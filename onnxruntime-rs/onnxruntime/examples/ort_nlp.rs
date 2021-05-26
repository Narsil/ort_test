use onnxruntime::environment::Environment;
use onnxruntime::ndarray::prelude::*;
use onnxruntime::ndarray::{IxDyn, OwnedRepr};
use onnxruntime::{session::Session, GraphOptimizationLevel, LoggingLevel};
use std::time::Instant;
use tokenizers::Tokenizer;

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
    // let start = Instant::now();
    session.inner_run().unwrap();
    // println!("Time elapsed in inner() is: {:?}", start.elapsed());
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
        // println!("past {:?}", past_key_values[0].shape());
        let (new_id, out_past_key_values) = run(
            session,
            input_ids,
            Array::from_shape_vec((1, attention_mask.len()), attention_mask.clone()).unwrap(),
            &past_key_values,
        );

        // println!("Output ids {:?}", output_ids);
        output_ids.push(new_id);
        input_ids = array![[new_id]];
        attention_mask.push(1);
        past_key_values = out_past_key_values;
    }
    Array::from_shape_vec((1, output_ids.len()), output_ids).unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = Tokenizer::from_file("tokenizer.json").unwrap();
    let environment = Environment::builder()
        .with_name("test")
        .with_log_level(LoggingLevel::Verbose)
        .build()?;
    let mut session = environment
        .new_session_builder()?
        .with_optimization_level(GraphOptimizationLevel::All)?
        // .with_number_threads(1)?
        .with_model_from_file("gpt2.onnx")?;
    let start = Instant::now();
    let ids: Vec<_> = tokenizer
        .encode("test", true)
        .unwrap()
        .get_ids()
        .into_iter()
        .map(|id| *id as i64)
        .collect();
    let input_ids = Array::from_shape_vec((1, ids.len()), ids).unwrap();
    let output_ids = generate(input_ids, 50, &mut session);
    let output_ids_u32 = output_ids
        .into_raw_vec()
        .into_iter()
        .map(|i| i as u32)
        .collect::<Vec<_>>();
    let output = tokenizer.decode(output_ids_u32, true).unwrap();
    println!(
        "Time elapsed in expensive_function() is: {:?}, output \n[{{\"generated_text\":\"{:?}\"}}]",
        start.elapsed(),
        output
    );
    Ok(())
}
