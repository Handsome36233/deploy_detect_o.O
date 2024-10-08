#![allow(clippy::manual_retain)]

use std::path::PathBuf;
use std::env;
use image::{imageops::FilterType, GenericImageView};
use ndarray::{s, Array, Axis};
use ort::{inputs, CUDAExecutionProvider, Session, SessionOutputs};
use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
use show_image::{event, AsImageView, WindowOptions};

#[derive(Debug, Clone, Copy)]
struct BoundingBox {
	x1: f32,
	y1: f32,
	x2: f32,
	y2: f32
}

fn intersection(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
	(box1.x2.min(box2.x2) - box1.x1.max(box2.x1)) * (box1.y2.min(box2.y2) - box1.y1.max(box2.y1))
}

fn union(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
	((box1.x2 - box1.x1) * (box1.y2 - box1.y1)) + ((box2.x2 - box2.x1) * (box2.y2 - box2.y1)) - intersection(box1, box2)
}


#[show_image::main]
fn main() -> ort::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: {} <model_path> <image_path>", args[0]);
        std::process::exit(1);
    }

    let model_path = PathBuf::from(&args[1]);
    let image_path = PathBuf::from(&args[2]);
	tracing_subscriber::fmt::init();

	ort::init()
		.with_execution_providers([CUDAExecutionProvider::default().build()])
		.commit()?;

	// 设置输入图像大小
	let input_width: u32 = 640;
	let input_height: u32 = 640;
	// 设置置信度阈值和 IoU 阈值
    let conf_threshold = 0.3;
    let iou_threshold = 0.5;
	let original_img = image::open(image_path).unwrap();
	let (img_width, img_height) = (original_img.width(), original_img.height());
	let img = original_img.resize_exact(input_width, input_height, FilterType::CatmullRom);
	// 创建大小为 (1, 3, 384, 640) 的输入数组
	let mut input = Array::zeros((1, 3, input_height as usize, input_width as usize));
	for pixel in img.pixels() {
		let x = pixel.0 as _;
		let y = pixel.1 as _;
		let [r, g, b, _] = pixel.2.0;
		input[[0, 0, y, x]] = (r as f32) / 255.;
		input[[0, 1, y, x]] = (g as f32) / 255.;
		input[[0, 2, y, x]] = (b as f32) / 255.;
	}

	let model = Session::builder()?.commit_from_file(model_path)?;

	// Run YOLOv8 inference
	let outputs: SessionOutputs = model.run(inputs!["images" => input.view()]?)?;
	let output = outputs["output0"].try_extract_tensor::<f32>()?.t().into_owned();

	let mut boxes = Vec::new();
	let output = output.slice(s![.., .., 0]);
	for row in output.axis_iter(Axis(1)) {
		let row: Vec<_> = row.iter().copied().collect();
		let (class_id, prob) = row
			.iter()
			// skip bounding box coordinates
			.skip(5)
			.enumerate()
			.map(|(index, value)| (index, *value))
			.reduce(|accum, row| if row.1 > accum.1 { row } else { accum })
			.unwrap();
        let conf = row[4];
		if prob*conf < conf_threshold {
			continue;
		}
        println!("class_id: {}, prob: {}", class_id, prob);

		let xc = row[0] / input_width as f32 * (img_width as f32);
		let yc = row[1] / input_height as f32 * (img_height as f32);
		let w = row[2] / input_width as f32 * (img_width as f32);
		let h = row[3] / input_height as f32 * (img_height as f32);
		boxes.push((
			BoundingBox {
				x1: xc - w / 2.,
				y1: yc - h / 2.,
				x2: xc + w / 2.,
				y2: yc + h / 2.
			},
			class_id,
			conf
		));
	}

	boxes.sort_by(|box1, box2| box2.2.total_cmp(&box1.2));
	let mut result = Vec::new();

	while !boxes.is_empty() {
		result.push(boxes[0]);
		boxes = boxes
			.iter()
			.filter(|box1| intersection(&boxes[0].0, &box1.0) / union(&boxes[0].0, &box1.0) < iou_threshold)
			.copied()
			.collect();
	}

	let mut dt = DrawTarget::new(img_width as _, img_height as _);

	for (bbox, label, _confidence) in result {
		let mut pb = PathBuilder::new();
		pb.rect(bbox.x1, bbox.y1, bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
		let path = pb.finish();
		let color = SolidSource { r: 0x80, g: 0x10, b: 0x40, a: 0x80 };
		dt.stroke(
			&path,
			&Source::Solid(color),
			&StrokeStyle {
				join: LineJoin::Round,
				width: 4.,
				..StrokeStyle::default()
			},
			&DrawOptions::new()
		);
	}

	let overlay: show_image::Image = dt.into();

	let window = show_image::context()
		.run_function_wait(move |context| -> Result<_, String> {
			let mut window = context
				.create_window(
					"ort + YOLOv5",
					WindowOptions {
						size: Some([img_width, img_height]),
						..WindowOptions::default()
					}
				)
				.map_err(|e| e.to_string())?;
			window.set_image("baseball", &original_img.as_image_view().map_err(|e| e.to_string())?);
			window.set_overlay("yolo", &overlay.as_image_view().map_err(|e| e.to_string())?, true);
			Ok(window.proxy())
		})
		.unwrap();

	for event in window.event_channel().unwrap() {
		if let event::WindowEvent::KeyboardInput(event) = event {
			if event.input.key_code == Some(event::VirtualKeyCode::Escape) && event.input.state.is_pressed() {
				break;
			}
		}
	}

	Ok(())
}
