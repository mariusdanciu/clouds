use std::{
    fs::File,
    io::{BufReader, Read},
    sync::Arc,
};

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
    },
    device::{Device, Queue},
    format::Format,
    image::{
        sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
        view::ImageView,
        Image, ImageCreateInfo, ImageType, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync,
};

use sync::GpuFuture;

use crate::{app::Texture, errors::AppError};

pub struct TextureLoader {}

impl TextureLoader {
    pub fn from_file_to_gpu(
        path: &str,
        device: Arc<Device>,
        queue: Arc<Queue>,
    ) -> Result<Texture, AppError> {
        let res = 128u32;

        let mut buf: Vec<u8> = Vec::with_capacity((res * res * res) as usize);
        let mut reader = BufReader::new(File::open(path)?);
        reader.read_to_end(&mut buf);

        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

        let image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim3d,
                format: Format::R8_UNORM,
                extent: [res, res, res],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                ..Default::default()
            },
            AllocationCreateInfo::default(),
        )?;

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

        let upload_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            buf,
        )?;

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo::default(),
        ));

        let mut builder = AutoCommandBufferBuilder::primary(
            command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )?;

        builder.copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            upload_buffer,
            image.clone(),
        ))?;

        let b = builder.build()?;

        sync::now(device.clone())
            .then_execute(queue.clone(), b)?
            .then_signal_fence_and_flush()?
            .wait(None)?;

        Ok(Texture {
            image_view,
            sampler,
        })
    }
}
