use crate::helper;
use enum_dispatch::enum_dispatch;

#[derive(Clone)]
pub enum EstimatorInit {
  SampleAverageInit {nb_levers : usize},
  ConstantStepInit {nb_levers : usize,
                    step : f64},
}

pub(crate) fn create_estimator(init_data : &EstimatorInit) -> EstimatorEnum {
  match *init_data {
    EstimatorInit::SampleAverageInit {nb_levers} =>
      SampleAverage::new(nb_levers).into(),
    EstimatorInit::ConstantStepInit {nb_levers, step} =>
      ConstantStep::new(nb_levers,step).into(),
  }
}

#[enum_dispatch]
#[derive(Clone)]
pub(crate) enum EstimatorEnum {
  SampleAverage,
  ConstantStep,
}

#[enum_dispatch(EstimatorEnum)]
pub(crate) trait Estimator : Send {
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

#[derive(Clone)]
pub(crate) struct SampleAverage {
  counter : Vec<f64>,
  estimates : Vec<f64>,
}

impl SampleAverage {

  pub(crate) fn new(nb_levers : usize) -> Self {
    SampleAverage {
      counter : vec![1.0;nb_levers],
      estimates : vec![0.0;nb_levers],
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
pub(crate) struct ConstantStep {
  step : f64,
  estimates : Vec<f64>,
}

impl ConstantStep {

  pub(crate) fn new(nb_levers : usize, step : f64) -> Self {
    ConstantStep {
      step,
      estimates : vec![0.0;nb_levers],
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
