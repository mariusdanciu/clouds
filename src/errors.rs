use std::io;

use thiserror::Error;
use vulkano::{
    LoadingError, Validated, ValidationError, VulkanError, buffer::AllocateBufferError, command_buffer::CommandBufferExecError, image::AllocateImageError, pipeline::layout::IntoPipelineLayoutCreateInfoError, swapchain::FromWindowError
};
use winit::{error::EventLoopError, raw_window_handle::HandleError};

#[derive(Debug, Error)]
pub enum AppError {

    #[error("Option error")]
    OptionError,

    #[error("IO error: {0}")]
    IOError(#[from] io::Error),

    #[error("Vulkano error: {0}")]
    Vulkan(#[from] vulkano::VulkanError),

    #[error("Vulkano error: {0}")]
    ValidateImage(#[from] Validated<AllocateImageError>),

    #[error("Vulkano error: {0}")]
    ValidatedBuffer(#[from] Validated<AllocateBufferError>),

    #[error("Vulkano error: {0}")]
    ValidatedVulkan(#[from] Validated<VulkanError>),

    #[error("Vulkano error: {0}")]
    ValidatedBoxVulkan(#[from] Box<ValidationError>),

    #[error("Vulkano error: {0}")]
    CommandBuffer(#[from] CommandBufferExecError),

    #[error("Vulkano error: {0}")]
    PipelineLayout(#[from] IntoPipelineLayoutCreateInfoError),

    #[error("Vulkano error: {0}")]
    Loading(#[from] LoadingError),

    #[error("Vulkano error: {0}")]
    Handle(#[from] HandleError),

    #[error("Vulkano error: {0}")]
    ValidatedHandle(#[from] Validated<HandleError>),
    
    #[error("winit error: {0}")]
    WinitError(#[from] winit::error::OsError),

    #[error("winit error: {0}")]
    WindowError(#[from] FromWindowError),

    #[error("winit error: {0}")]
    EventLoopError(#[from] EventLoopError),
    
}
