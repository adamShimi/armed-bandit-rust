use crate::helper;
use enum_dispatch::enum_dispatch;

#[derive(Clone)]
pub enum EstimatorInit {
  SampleAverageInit {nb_levers : usize},
  ConstantStepInit {nb_levers : usize,
                    step : f64},
}

#[enum_dispatch]
#[derive(Clone)]
pub enum EstimatorEnum {
  SampleAverage,
  ConstantStep,
}

#[enum_dispatch(EstimatorEnum)]
pub trait Estimator : Send {
  // Give the current estimate of the required lever.
  fn estimate(&self, lever : usize) -> f64;

  // Update the estimate of the lever according to
  // the reward.
  fn update(&mut self, lever : usize, reward : f64);

  // Give all estimates.
  fn all(&self, nb_levers : usize) -> Vec<f64> {
    (0..nb_levers).map(|x| self.estimate(x))
                  .collect()
  }

  // Find the list of levers with the best estimate.
  fn optimal(&self, nb_levers : usize) -> Vec<usize> {
    helper::indices_max(&self.all(nb_levers))
  }
}

pub fn create_estimator(init_data : EstimatorInit) -> EstimatorEnum {
  match init_data {
    init @ EstimatorInit::SampleAverageInit {..} => SampleAverage::new(init).into(),
    init @ EstimatorInit::ConstantStepInit {..} => ConstantStep::new(init).into(),
  }
}

#[derive(Clone)]
pub struct SampleAverage {
  pub counter : Vec<f64>,
  pub estimates : Vec<f64>,
}

impl SampleAverage {

  pub fn new(init_data : EstimatorInit) -> Self {
    match init_data {
      EstimatorInit::SampleAverageInit {nb_levers} =>
        SampleAverage {
          counter : vec![1.0;nb_levers],
          estimates : vec![0.0;nb_levers],
        },
      _ => {panic!("Cannot create SampleAverage from ConstantStepInit");},
    }
  }
}

impl Estimator for SampleAverage {

  fn estimate(&self, lever : usize) -> f64 {
    self.estimates[lever]
  }

  fn update(&mut self, lever : usize, reward : f64) {
    self.estimates[lever] =
      self.estimates[lever] + (reward - self.estimates[lever])/self.counter[lever];
    self.counter[lever] += 1.0;
  }
}


#[derive(Clone)]
pub struct ConstantStep {
  pub step : f64,
  pub estimates : Vec<f64>,
}

impl ConstantStep {

  pub fn new(init_data : EstimatorInit) -> Self {
    match init_data {
      EstimatorInit::ConstantStepInit {nb_levers, step} =>
        ConstantStep {
          step,
          estimates : vec![0.0;nb_levers],
        },
      _ => {panic!("Cannot create ConstantStep from SampleAverageInit");},
    }
  }
}

impl Estimator for ConstantStep {

  fn estimate(&self, lever : usize) -> f64 {
    self.estimates[lever]
  }

  fn update(&mut self, lever : usize, reward : f64) {
    self.estimates[lever] =
      self.estimates[lever] + self.step*(reward - self.estimates[lever]);
  }
}
