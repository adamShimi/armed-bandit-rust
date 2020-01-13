use crate::helper;

use std::collections::HashSet;
use std::iter::FromIterator;

use rand::Rng;
use rand_distr::{Normal, Distribution};

pub enum BanditInit {
  StationaryInit { nb_levers : usize,
                   init_vals : (f64,f64)
                 },
  NonStationaryInit { nb_levers : usize,
                      init_vals : (f64,f64),
                      walk : (f64,f64)
                    },
}

pub trait Bandit : Clone + Send {
  // Get reward from a lever.
  fn use_lever<T: Rng>(&mut self, lever : usize, rng: &mut T) -> f64;

  // Get set of optimal levers.
  fn is_optimal(&self, lever : usize) -> bool;
}

// Implementatio of a stationary bandit instance, where
// nb_levers levers are initialized at according to a normal distribution.
#[derive(Clone)]
pub struct BanditStationary {
  levers : Vec<Normal<f64>>,
  optimals : HashSet<usize>,
}

impl BanditStationary {

  pub fn new<T: Rng>(nb_levers : usize, init : (f64,f64),rng : &mut T) -> Self {
    let init_distrib = Normal::new(init.0,init.1).unwrap();
    let (levers,optimals) : (Vec<Normal<f64>>,Vec<f64>) =
      (0..nb_levers).map(|_| {
                      let mean = init_distrib.sample(rng);
                      (Normal::new(mean, init.1).unwrap(),mean)
                    })
                    .unzip();
    BanditStationary {
      levers,
      optimals : HashSet::from_iter(helper::indices_max(&optimals)),
    }
  }
}

impl Bandit for BanditStationary {

  fn use_lever<T: Rng>(&mut self, lever: usize, rng: &mut T) -> f64 {
    self.levers[lever].sample(rng)
  }

  fn is_optimal(&self, lever : usize) -> bool {
    self.optimals.contains(&lever)
  }

}




// Implementation of a nonstationary bandit problems, where
// levers start with the same mean and move according to a random
// walk at each step.
#[derive(Clone)]
pub struct BanditNonStationary {
  levers : Vec<f64>,
  // Only one std, because only the means move according to
  // the random walk; the standard deviation stays the same.
  std : f64,
  walks : Vec<Normal<f64>>,
  optimals : HashSet<usize>,
}

impl BanditNonStationary {

  pub fn new(nb_levers : usize, init : (f64,f64), walk : (f64,f64)) -> Self {
    BanditNonStationary {
      levers : vec![init.0; nb_levers],
      std : init.1,
      walks : (0..nb_levers).map(|_| Normal::new(walk.0,walk.1).unwrap())
                            .collect(),
      optimals : (0..nb_levers).collect(),
    }
  }

  fn update<T: Rng>(&mut self, rng : &mut T) {
    self.levers =
      self.levers.iter()
           .zip(self.walks.iter())
           .map(|(lever,walk)| lever + walk.sample(rng))
           .collect();
    self.optimals.clear();
    self.optimals = HashSet::from_iter(helper::indices_max(&self.levers));
  }
}

impl Bandit for BanditNonStationary {

  fn use_lever<T: Rng>(&mut self, lever: usize, rng: &mut T) -> f64 {
    let result = Normal::new(self.levers[lever],self.std)
             .unwrap()
             .sample(rng);
    self.update(rng);
    result

  }

  fn is_optimal(&self, lever : usize) -> bool {
    self.optimals.contains(&lever)
  }

}
