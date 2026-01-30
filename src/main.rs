use app::App;
use winit::event_loop::EventLoop;

use crate::errors::AppError;

mod app;
mod camera;
mod computetex;
mod errors;
mod shaders;
mod texture_loader;

fn main() -> Result<(), AppError> {
    let event_loop = EventLoop::new()?;

    let mut app = App::new(&event_loop)?;

    event_loop.run_app(&mut app);
    Ok(())
}
