use crate::helper;

use rand::Rng;
use rand::prelude::IteratorRandom;

pub enum PolicyInit {
  EGreedyInit {nb_levers : usize,
               expl_proba : f64,
               est : estimators::EstimatorInit},
  UCBInit {nb_levers : usize,
           est : estimators::EstimatorInit},
}

pub trait Policy : Clone + Send {
  // Choose the action: either a lever for exploiting
  // or the exploring option.
  fn decide<V: Rng>(&self, rng: &mut V) -> usize;

  // Update its values based on the result of the
  // step.
  fn update(&mut self, lever : usize, reward : f64);
}

#[derive(Clone)]
pub struct EGreedy {
  nb_levers : usize,
  expl_proba : f64,
  estimator : Box<dyn estimators::Estimator>,
}

impl EGreedy {

  pub fn new(nb_levers : usize,
             expl_proba: f64,
             estimator : Box<dyn estimators::Estimator>) -> Self {
    EGreedy {
      nb_levers,
      expl_proba,
      estimator,
    }
  }


  fn explore<V: Rng>(&self, rng : &mut V) -> usize {
    rng.gen_range(0,self.nb_levers)
  }
}

impl Policy for EGreedy {

  fn decide<V: Rng>(&self, rng: &mut V) -> usize {
    if rng.gen_bool(self.expl_proba) {
      self.explore(rng)
    } else {
      *self.estimator.optimal(self.nb_levers)
                     .iter()
                     .choose(rng)
                     .unwrap()
    }
  }

  // Update its values based on the result of the
  // step.
  fn update(&mut self, lever : usize, reward : f64) {
    self.estimator.update(lever,reward);
  }
}

#[derive(Clone)]
pub struct UCB {
  nb_levers : usize,
  time : f64,
  counts : Vec<f64>,
  estimator : Box<dyn estimators::Estimator>,
}

impl UCB {

  pub fn new(nb_levers : usize, estimator : Box<dyn estimators::Estimator>) -> Self {
    UCB {
      nb_levers,
      time : 0.0,
      counts : vec![0.0;nb_levers],
      estimator,
    }
  }
}

impl Policy for UCB {

  fn decide<V: Rng>(&self, rng: &mut V) -> usize {
    let est_counts : Vec<f64> =
      self.estimator.all(self.nb_levers)
                    .iter()
                    .zip(self.counts.iter())
                    .map( |x| x.0 + (2.0*self.time.log(2.0) / *x.1).sqrt())
                    .collect();
    *helper::indices_max(&est_counts)
            .iter()
            .choose(rng)
            .unwrap()
  }

  // Update its values based on the result of the
  // step.
  fn update(&mut self, lever : usize, reward : f64) {
    self.estimator.update(lever,reward);
  }
}

pub mod estimators {

  use crate::helper;

  use dyn_clone::DynClone;

  pub enum EstimatorInit {
    SampleAverageInit {nb_levers : usize},
    ConstantStepInit {nb_levers : usize,
                      step : f64},
  }

  pub trait Estimator : DynClone + Send {
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

  dyn_clone::clone_trait_object!(Estimator);

  #[derive(Clone)]
  pub struct SampleAverage {
    pub counter : Vec<f64>,
    pub estimates : Vec<f64>,
  }

  impl SampleAverage {

    pub fn new(nb_levers : usize) -> Self {
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
  pub struct ConstantStep {
    pub step : f64,
    pub estimates : Vec<f64>,
  }

  impl ConstantStep {

    pub fn new(nb_levers : usize, step : f64) -> Self {
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
}
