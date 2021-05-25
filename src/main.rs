use actix_web::{middleware::Logger, web, App, HttpResponse, HttpServer};
use onnxruntime::environment::Environment;
use onnxruntime::ndarray::prelude::*;
use onnxruntime::ndarray::{IxDyn, OwnedRepr};
use onnxruntime::{session::Session, GraphOptimizationLevel, LoggingLevel};
use serde::Deserialize;
use std::sync::Mutex;
use std::time::Instant;
use tokenizers::Tokenizer;

// This struct represents state
struct AppState {
    session: Mutex<Session>,
    tokenizer: Mutex<Tokenizer>,
}

#[derive(Deserialize)]
struct Parameters {
    n_tokens: usize,
}
impl Default for Parameters {
    fn default() -> Parameters {
        Parameters { n_tokens: 32 }
    }
}

#[derive(Deserialize)]
struct Input {
    inputs: String,
    #[serde(default)]
    parameters: Parameters,
}

async fn index(data: web::Data<AppState>, input: web::Json<Input>) -> HttpResponse {
    let start = Instant::now();
    let input_ids = {
        let tokenizer = &data.tokenizer.lock().unwrap();
        let ids: Vec<_> = tokenizer
            .encode(input.inputs.clone(), true)
            .unwrap()
            .get_ids()
            .into_iter()
            .map(|id| *id as i64)
            .collect();
        Array::from_shape_vec((1, ids.len()), ids).unwrap()
    };
    let output_ids = {
        let session = &mut data.session.lock().unwrap();
        generate(input_ids, input.parameters.n_tokens, session)
    };
    let output_ids_u32 = output_ids
        .into_raw_vec()
        .into_iter()
        .map(|i| i as u32)
        .collect::<Vec<_>>();
    let output = {
        let tokenizer = &data.tokenizer.lock().unwrap();
        tokenizer.decode(output_ids_u32, true).unwrap()
    };

    HttpResponse::Ok()
        .content_type("application/json")
        .header("x-compute-time", format!("{:?}", start.elapsed()))
        .body(format!("data: {:?}", output))
}

fn run<'s, 'm, 't>(
    session: &'s mut Session,
    input_ids: ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
    attention_mask: ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
    past_key_values: &[ArrayBase<OwnedRepr<f32>, IxDyn>],
) -> (i64, Vec<ArrayBase<OwnedRepr<f32>, IxDyn>>)
where
    'm: 't,
    's: 'm,
{
    // println!("input ids {:?}", input_ids);
    // println!("attention mask {:?}", attention_mask);
    session.feed(input_ids).unwrap();
    // println!("{:?}", past_key_values[0]);
    past_key_values.into_iter().for_each(|past_key_value| {
        session.feed(past_key_value.to_owned()).unwrap();
    });
    session.feed(attention_mask).unwrap();
    session.inner_run().unwrap();
    let new_id = {
        let logits: ArrayBase<OwnedRepr<f32>, IxDyn> = session.read().unwrap().to_owned();
        // println!("Logits {:?}", logits);
        let new_id = argmax(&logits, 0) as i64;
        new_id
    };
    // let start = Instant::now();
    let out_past_key_values = (0..24)
        .map(|_| session.read().unwrap().to_owned())
        .collect();
    // println!("Time elapsed in inner() is: {:?}", start.elapsed());

    (new_id, out_past_key_values)
}

fn argmax<T>(matrix: &ArrayBase<OwnedRepr<T>, IxDyn>, axis: usize) -> usize
where
    T: std::cmp::PartialOrd + Copy + std::fmt::Debug,
    // ViewRepr<T>: RawData,
    // <ViewRepr<T> as RawData>::Elem: std::cmp::PartialOrd + Copy,
{
    for (_, row) in matrix.axis_iter(Axis(axis)).enumerate() {
        let (max_idx, _) =
            row.iter()
                .enumerate()
                .fold((0, row[[0, 0]]), |(idx_max, val_max), (idx, val)| {
                    if &val_max > val {
                        (idx_max, val_max)
                    } else {
                        (idx, *val)
                    }
                });
        return max_idx;
    }
    return 100000;
}

fn generate(
    mut input_ids: ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>>,
    tokens: usize,
    session: &mut Session,
) -> ArrayBase<OwnedRepr<i64>, Dim<[usize; 2]>> {
    let n = input_ids.shape()[1];
    let mut output_ids = input_ids.clone().into_raw_vec();
    let mut attention_mask = vec![1i64; n];
    let mut past_key_values: Vec<_> = (0..24)
        .map(|_| Array::<f32, _>::zeros(IxDyn(&[1, 12, 0, 64])))
        .collect();

    for _ in 0..tokens - 1 {
        let (new_id, out_past_key_values) = run(
            session,
            input_ids,
            Array::from_shape_vec((1, attention_mask.len()), attention_mask.clone()).unwrap(),
            &past_key_values,
        );
        // println!("Loop1 {:?}", start.elapsed());
        let start = Instant::now();

        output_ids.push(new_id);
        input_ids = array![[new_id]];
        attention_mask.push(1);
        past_key_values = out_past_key_values;
        println!("Loop {:?}", start.elapsed());
    }
    Array::from_shape_vec((1, output_ids.len()), output_ids).unwrap()
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    std::env::set_var("RUST_LOG", "actix_web=info");
    env_logger::init();
    HttpServer::new(|| {
        println!("New here");
        let environment = Environment::builder()
            .with_name("app")
            .with_log_level(LoggingLevel::Verbose)
            .build()
            .unwrap();
        let session = environment
            .new_session_builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::All)
            .unwrap()
            // .with_number_threads(1)
            // .unwrap()
            .with_model_from_file("gpt2.onnx")
            .unwrap();
        let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
        let state = web::Data::new(AppState {
            // app_name: Mutex::new(String::from("Actix-web")),
            // environment,
            session: Mutex::new(session),
            tokenizer: Mutex::new(tokenizer),
        });
        println!("Created state");
        App::new()
            .wrap(Logger::default())
            .app_data(state)
            .route("/", web::post().to(index))
    })
    .workers(1)
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
