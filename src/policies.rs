use crate::estimators::{Estimator,EstimatorInit,EstimatorEnum, create_estimator};
use crate::helper;

use rand::Rng;
use rand::prelude::IteratorRandom;
use enum_dispatch::enum_dispatch;

#[derive(Clone)]
pub enum PolicyInit<'a> {
  EGreedyInit {nb_levers : usize,
               expl_proba : f64,
               est : &'a EstimatorInit},
  UCBInit {nb_levers : usize,
           step : f64,
           est : &'a EstimatorInit},
}

pub(crate) fn create_policy(init_data : &PolicyInit) -> PolicyEnum {
  match init_data {
    &PolicyInit::EGreedyInit {nb_levers,expl_proba,est} =>
      EGreedy::new(nb_levers,expl_proba,est).into(),
    &PolicyInit::UCBInit {nb_levers,step, est} =>
      UCB::new(nb_levers,step,est).into(),
  }
}

#[enum_dispatch]
#[derive(Clone)]
pub(crate) enum PolicyEnum {
  EGreedy,
  UCB,
}

#[enum_dispatch(PolicyEnum)]
pub(crate) trait Policy : Clone + Send {
  // Choose the action: either a lever for exploiting
  // or the exploring option.
  fn decide<V: Rng>(&self, rng: &mut V) -> usize;

  // Update its values based on the result of the
  // step.
  fn update(&mut self, lever : usize, reward : f64);
}

#[derive(Clone)]
pub(crate) struct EGreedy {
  nb_levers : usize,
  expl_proba : f64,
  estimator : EstimatorEnum,
}

impl EGreedy {

  pub(crate) fn new(nb_levers : usize, expl_proba : f64, est : &EstimatorInit) -> Self {
    EGreedy {
      nb_levers,
      expl_proba,
      estimator : create_estimator(est).into()
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
pub(crate) struct UCB {
  nb_levers : usize,
  step : f64,
  time : f64,
  counts : Vec<f64>,
  estimator : EstimatorEnum,
}

impl UCB {

  pub(crate) fn new(nb_levers : usize, step : f64, est : &EstimatorInit) -> Self {
    UCB {
      nb_levers,
      step,
      time : 0.0,
      counts : vec![0.0;nb_levers],
      estimator : create_estimator(est).into()
    }
  }
}

impl Policy for UCB {

  fn decide<V: Rng>(&self, rng: &mut V) -> usize {
    let est_counts : Vec<f64> =
      self.estimator.all(self.nb_levers)
                    .iter()
                    .zip(self.counts.iter())
                    .map( |x| x.0 + self.step*(self.time.ln() / *x.1).sqrt())
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
