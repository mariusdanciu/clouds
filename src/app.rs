use glam::{vec3, Vec2, Vec3};
use image::{DynamicImage, ImageReader, ImageResult};
use std::{sync::Arc, time::Instant};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Queue,
        QueueCreateInfo, QueueFlags,
    },
    image::{sampler::Sampler, view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, VulkanError, VulkanLibrary,
};

use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, StartCause, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

use crate::shaders::vertex;
use crate::{
    camera::{Camera, CameraEvent},
    computetex::ComputeTex,
};
use crate::{errors::AppError, shaders::compute};
use crate::{shaders::fragment, texture_loader::TextureLoader};

pub struct Texture {
    pub image_view: Arc<ImageView>,
    pub sampler: Arc<Sampler>,
}

pub struct App {
    instance: Arc<Instance>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    texture: Arc<Texture>,
    vertex_buffer: Subbuffer<[MyVertex]>,
    render_ctx: Option<RenderContext>,
    camera: Camera,
    mouse_pressed: bool,
    last_mouse_pos: Vec2,
    cam_rotation: Option<Vec2>,
    cam_up: bool,
    cam_down: bool,
    cam_left: bool,
    cam_right: bool,
    frame_time: Instant,
    fps: u32,
    ups: u32,
    timer: Instant,
    i_time: Instant,
}

const NANOS: f32 = 1000000000. / 60.;

struct RenderContext {
    window: Arc<Window>,
    swapchain: Arc<Swapchain>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    viewport: Viewport,
    recreate_swapchain: bool,
    previous_frame_end: Option<Box<dyn GpuFuture>>,
}

impl App {
    fn texture() -> ImageResult<DynamicImage> {
        ImageReader::open("./resources/out.png")?.decode()
    }

    pub fn new(event_loop: &EventLoop<()>) -> Result<Self, AppError> {
        let library = VulkanLibrary::new()?;

        let required_extensions = Surface::required_extensions(event_loop)?;

        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                // Enable enumerating devices that use non-conformant Vulkan implementations.
                // (e.g. MoltenVK)
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )?;

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()?
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.presentation_support(i as u32, event_loop).unwrap()
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| {
                // We assign a lower score to device types that are likely to be faster/better.
                match p.properties().device_type {
                    PhysicalDeviceType::DiscreteGpu => 0,
                    PhysicalDeviceType::IntegratedGpu => 1,
                    PhysicalDeviceType::VirtualGpu => 2,
                    PhysicalDeviceType::Cpu => 3,
                    PhysicalDeviceType::Other => 4,
                    _ => 5,
                }
            })
            .expect("no suitable physical device found");

        // Some little debug infos.
        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],

                ..Default::default()
            },
        )?;

        let queue = queues.next().ok_or(AppError::OptionError)?;

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(), // No pool_sizes here in 0.35.2
        ));

        let vertices = [
            MyVertex {
                position: [-1.0, -1.0],
            },
            MyVertex {
                position: [-1.0, 1.0],
            },
            MyVertex {
                position: [1.0, -1.0],
            },
            MyVertex {
                position: [1.0, 1.0],
            },
            MyVertex {
                position: [1.0, -1.0],
            },
            MyVertex {
                position: [-1.0, 1.0],
            },
        ];
        let vertex_buffer = Buffer::from_iter(
            memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            vertices,
        )?;

        let rcx = None;

        let camera = Camera::new_with_pos(Vec3::new(-0.5, 3., 8.0), Vec3::new(0., -1., -5.));

        // let tex = ComputeTex {
        //     width: 256,
        //     height: 256,
        //     depth: 256,
        // };

        let texture = TextureLoader::from_file_to_gpu(
            "./resources/cloud_noise.raw",
            device.clone(),
            queue.clone(),
        )?;

        // let texture = tex.run(
        //     device.clone(),
        //     descriptor_set_allocator.clone(),
        //     queue.clone(),
        // )?;

        let texture = Arc::new(texture);

        Ok(App {
            instance,
            device,
            queue,
            command_buffer_allocator,
            descriptor_set_allocator,
            texture,
            vertex_buffer,
            render_ctx: rcx,
            camera,
            mouse_pressed: false,
            last_mouse_pos: Vec2::ZERO,
            cam_rotation: None,
            cam_down: false,
            cam_left: false,
            cam_right: false,
            cam_up: false,
            frame_time: Instant::now(),
            fps: 0u32,
            ups: 0u32,
            timer: Instant::now(),
            i_time: Instant::now(),
        })
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) -> Result<(), AppError> {
        let window = Arc::new(event_loop.create_window(Window::default_attributes())?);
        let surface = Surface::from_window(self.instance.clone(), window.clone())?;
        let window_size = window.inner_size();

        let (swapchain, images) = {
            let surface_capabilities = self
                .device
                .physical_device()
                .surface_capabilities(&surface, Default::default())?;

            let (image_format, _) = self
                .device
                .physical_device()
                .surface_formats(&surface, Default::default())?[0];

            Swapchain::new(
                self.device.clone(),
                surface,
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count.max(2),
                    image_format,
                    image_extent: window_size.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .into_iter()
                        .next()
                        .ok_or(AppError::OptionError)?,
                    present_mode: vulkano::swapchain::PresentMode::Fifo,
                    ..Default::default()
                },
            )?
        };

        let render_pass = vulkano::single_pass_renderpass!(
            self.device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(),
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {},
            },
        )?;

        let framebuffers = window_size_dependent_setup(&images, &render_pass)?;
        let pipeline = {
            let vs = vertex::load(self.device.clone())?
                .entry_point("main")
                .ok_or(AppError::OptionError)?;
            let fs = fragment::load(self.device.clone())?
                .entry_point("main")
                .ok_or(AppError::OptionError)?;

            let vertex_input_state = MyVertex::per_vertex().definition(&vs)?;

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                self.device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(self.device.clone())?,
            )?;

            let subpass = Subpass::from(render_pass.clone(), 0).ok_or(AppError::OptionError)?;

            GraphicsPipeline::new(
                self.device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )?
        };

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: window_size.into(),
            depth_range: 0.0..=1.0,
        };

        let recreate_swapchain = false;

        let previous_frame_end = Some(sync::now(self.device.clone()).boxed());

        self.render_ctx = Some(RenderContext {
            window,
            swapchain,
            render_pass,
            framebuffers,
            pipeline,
            viewport,
            recreate_swapchain,
            previous_frame_end,
        });

        Ok(())
    }

    fn redraw(&mut self) -> Result<(), AppError> {
        let mut rcx = self.render_ctx.as_mut().ok_or(AppError::OptionError)?;
        let window_size = rcx.window.inner_size();

        if window_size.width == 0 || window_size.height == 0 {
            return Ok(());
        }
        if self.fps == 0 {
            rcx.previous_frame_end
                .as_mut()
                .ok_or(AppError::OptionError)?
                .cleanup_finished();
        }
        if rcx.recreate_swapchain {
            let (new_swapchain, new_images) = rcx
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: window_size.into(),
                    ..rcx.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            rcx.swapchain = new_swapchain;
            rcx.framebuffers = window_size_dependent_setup(&new_images, &rcx.render_pass)?;
            rcx.viewport.extent = window_size.into();
            rcx.recreate_swapchain = false;
        }

        let (image_index, suboptimal, acquire_future) =
            match acquire_next_image(rcx.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    rcx.recreate_swapchain = true;
                    return Ok(());
                }
                Err(e) => Err(e)?,
            };

        if suboptimal {
            rcx.recreate_swapchain = true;
        }

        let millis = self.timer.elapsed().as_millis();

        self.ups += 1;

        if millis > 1000 {
            self.timer = Instant::now();
            rcx.window
                .set_title(format!("FPS {} UPS {}", self.fps, self.ups).as_str());
            self.fps = 0;
            self.ups = 0;
        }

        let t = self.i_time.elapsed().as_millis() as u32;

        let pc_screen = fragment::AppData {
            time: (t).into(),
            screen: [(window_size.width as f32), (window_size.height as f32)].into(),
            cam_position: self.camera.position.to_array().into(),
            cam_uu: self.camera.uu.to_array().into(),
            cam_vv: self.camera.vv.to_array().into(),
            cam_ww: self.camera.ww.to_array().into(),
            materials: [
                fragment::Material {
                    volumetric: 0,
                    specular: 2.9.into(),
                    shininess: 320.0.into(),
                    roughness: 0.8.into(),
                    diffuse: 0.9.into(),
                    color: vec3(0.7, 0.0, 0.0).to_array().into(),
                }
                .into(),
                fragment::Material {
                    volumetric: 0,
                    specular: 0.5.into(),
                    shininess: 80.0.into(),
                    roughness: 0.8.into(),
                    diffuse: 1.1.into(),
                    color: vec3(0.9, 0.9, 0.8).to_array().into(),
                }
                .into(),
                fragment::Material {
                    volumetric: 1,
                    specular: 0.0.into(),
                    shininess: 0.0.into(),
                    roughness: 0.0.into(),
                    diffuse: 0.0.into(),
                    color: vec3(0.8, 0.8, 0.8).to_array().into(),
                }
                .into(),
            ],
        };

        let layout = rcx.pipeline.layout().clone();

        let ds_layout = layout
            .set_layouts()
            .get(0) // Must match the 'set = 0' in your shader
            .expect("Pipeline must have a descriptor set layout at index 0")
            .clone();

        let set = DescriptorSet::new(
            self.descriptor_set_allocator.clone(),
            ds_layout.clone(),
            [WriteDescriptorSet::image_view_sampler(
                0,
                self.texture.image_view.clone(),
                self.texture.sampler.clone(),
            )],
            [],
        )?;

        let mut builder = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(
                        rcx.framebuffers[image_index as usize].clone(),
                    )
                },
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )?
            .set_viewport(0, [rcx.viewport.clone()].into_iter().collect())?
            .push_constants(layout.clone(), 0, pc_screen)?
            .bind_descriptor_sets(PipelineBindPoint::Graphics, layout.clone(), 0, set.clone())?
            .bind_pipeline_graphics(rcx.pipeline.clone())?
            .bind_vertex_buffers(0, self.vertex_buffer.clone())?;

        // We add a draw command.
        unsafe { builder.draw(self.vertex_buffer.len() as u32, 1, 0, 0) }?;

        builder.end_render_pass(Default::default())?;

        let command_buffer = builder.build()?;

        let sc_info =
            SwapchainPresentInfo::swapchain_image_index(rcx.swapchain.clone(), image_index);

        let future = rcx
            .previous_frame_end
            .take()
            .ok_or(AppError::OptionError)?
            .join(acquire_future)
            .then_execute(self.queue.clone(), command_buffer)?
            .then_swapchain_present(self.queue.clone(), sc_info)
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                rcx.previous_frame_end = Some(future.boxed());
                self.fps += 1;
            }
            Err(VulkanError::OutOfDate) => {
                rcx.recreate_swapchain = true;
                rcx.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                Err(e)?;
            }
        }
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        match self.resumed(event_loop) {
            Ok(_) => (),
            Err(e) => eprintln!("{}", e),
        }
    }

    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        let elapsed = self.frame_time.elapsed();
        let ts = elapsed.as_secs_f32();

        let mut events: Vec<CameraEvent> = vec![];
        if self.cam_up {
            events.push(CameraEvent::Up)
        }
        if self.cam_down {
            events.push(CameraEvent::Down)
        }
        if self.cam_left {
            events.push(CameraEvent::Left)
        }
        if self.cam_right {
            events.push(CameraEvent::Right)
        }
        if let Some(delta) = self.cam_rotation {
            events.push(CameraEvent::RotateXY { delta })
        }
        if !events.is_empty() {
            self.camera.update(&events, ts);
        }
        self.cam_rotation = None;
        self.frame_time = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let rcx = self.render_ctx.as_mut().unwrap();

        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => {
                event_loop.exit();
            }

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key,
                        state,
                        repeat: false,
                        ..
                    },
                ..
            } => match physical_key {
                PhysicalKey::Code(KeyCode::KeyW) => {
                    self.cam_up = state.is_pressed();
                }
                PhysicalKey::Code(KeyCode::KeyS) => {
                    self.cam_down = state.is_pressed();
                }
                PhysicalKey::Code(KeyCode::KeyA) => {
                    self.cam_left = state.is_pressed();
                }
                PhysicalKey::Code(KeyCode::KeyD) => {
                    self.cam_right = state.is_pressed();
                }
                _ => {}
            },

            WindowEvent::CursorMoved {
                device_id,
                position,
            } => {
                let pos = Vec2::new(position.x as f32, position.y as f32);
                if self.mouse_pressed {
                    if self.last_mouse_pos == Vec2::ZERO {
                        self.last_mouse_pos = pos;
                    }
                    let delta = (pos - self.last_mouse_pos) * 0.05;
                    self.last_mouse_pos = pos;
                    if delta.x != 0.0 || delta.y != 0.0 {
                        self.cam_rotation = Some(delta);
                    }
                }
            }
            WindowEvent::MouseInput {
                device_id,
                state,
                button,
            } => match self.render_ctx.as_mut() {
                Some(ctx) => {
                    self.mouse_pressed = state.is_pressed();
                    ctx.window.set_cursor_visible(false);

                    if !self.mouse_pressed {
                        ctx.window.set_cursor_visible(true);
                        self.cam_rotation = None;
                        self.last_mouse_pos = Vec2::ZERO;
                    };
                }
                None => {
                    eprintln!("No render context yet");
                }
            },
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                rcx.recreate_swapchain = true;
            }
            WindowEvent::RedrawRequested => match self.redraw() {
                Ok(_) => (),
                Err(e) => eprintln!("{}", e),
            },
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        match self.render_ctx.as_mut() {
            Some(ctx) => {
                ctx.window.request_redraw();
            }
            None => {
                eprintln!("No context yet");
            }
        }
    }
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct MyVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

fn window_size_dependent_setup(
    images: &[Arc<Image>],
    render_pass: &Arc<RenderPass>,
) -> Result<Vec<Arc<Framebuffer>>, AppError> {
    let r = images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();

            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    Ok(r)
}
