use std::sync::Arc;

use vulkano::{
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    format::Format,
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
        Image, ImageCreateInfo, ImageType, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    sync::{self, GpuFuture},
};

use crate::{app::Texture, errors::AppError, shaders::compute};

pub struct ComputeTex {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl ComputeTex {
    pub fn run(
        &self,
        device: Arc<Device>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
        queue: Arc<Queue>,
    ) -> Result<Texture, AppError> {
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim3d,
                format: Format::R8G8B8A8_UNORM,
                extent: [self.width, self.height, self.depth],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED | ImageUsage::STORAGE,

                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )?;

        let cs = compute::load(device.clone())?
            .entry_point("main")
            .ok_or(AppError::OptionError)?;

        let stage = PipelineShaderStageCreateInfo::new(cs);
        let layout = PipelineLayout::new(
            device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                .into_pipeline_layout_create_info(device.clone())?,
        )?;

        let compute_pipeline = ComputePipeline::new(
            device.clone(),
            None,
            ComputePipelineCreateInfo::stage_layout(stage, layout),
        )?;

        let pipeline_layout = compute_pipeline.layout();
        let descriptor_set_layouts = pipeline_layout.set_layouts();

        let descriptor_set_layout_index = 0;
        let descriptor_set_layout = descriptor_set_layouts
            .get(descriptor_set_layout_index)
            .ok_or(AppError::OptionError)?;

        let image_view = ImageView::new_default(image.clone())?;

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )?;

        let descriptor_set = DescriptorSet::new(
            descriptor_set_allocator,
            descriptor_set_layout.clone(),
            [WriteDescriptorSet::image_view(0, image_view.clone())],
            [],
        )?;

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator,
            queue.clone().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        let work_group_counts = [self.width / 4, self.height / 4, self.depth];

        command_buffer_builder
            .bind_pipeline_compute(compute_pipeline.clone())?
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                descriptor_set_layout_index as u32,
                descriptor_set,
            )?;

        unsafe {
            command_buffer_builder.dispatch(work_group_counts)?;
        };

        let command_buffer = command_buffer_builder.build()?;
        let future = sync::now(device.clone())
            .then_execute(queue.clone(), command_buffer)?
            .then_signal_fence_and_flush()?;

        println!("Done computing");
        Ok(Texture {
            image_view: image_view,
            sampler: sampler,
        })
    }
}
