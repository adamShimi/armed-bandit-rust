pub mod problems {

  use rand_distr::{Normal, Distribution};

  pub trait Bandit {
    // Get reward from a lever.
    fn use_lever(&mut self, lever : usize) -> f64;
  }


  // Implementatio of a stationary bandit instance, where
  // nb_levers levers are initialized at according to a normal distribution.
  pub struct BanditStationary {
    levers : Vec<Normal<f64>>,
  }

  impl BanditStationary {

    pub fn new( nb_levers : usize, init : (f64,f64)) -> Self {
      let init_distrib = Normal::new(init.0,init.1).unwrap();
      BanditStationary {
        levers : (0..nb_levers).map(|_|
                     Normal::new(init_distrib.sample(&mut rand::thread_rng()),
                           init.1)
                         .unwrap()
                   )
                   .collect()
      }
    }
  }

  impl Bandit for BanditStationary {

    fn use_lever(&mut self, lever: usize) -> f64 {
      self.levers[lever].sample(&mut rand::thread_rng())
    }

  }




  // Implementation of a nonstationary bandit problems, where
  // levers start with the same mean and move according to a random
  // walk at each step.
  pub struct BanditNonStationary {
    levers : Vec<f64>,
    // Only one std, because only the means move according to
    // the random walk; the standard deviation stays the same.
    std : f64,
    walks : Vec<Normal<f64>>,
  }

  impl BanditNonStationary {

    pub fn new(nb_levers : usize, init : (f64,f64), walk : (f64,f64)) -> Self {
      BanditNonStationary {
        levers : vec![init.0; nb_levers],
        std : init.1,
        walks : (1..nb_levers).map(|_| Normal::new(walk.0,walk.1).unwrap())
                   .collect(),
      }
    }

    fn update(&mut self) {
      self.levers =
        self.levers.iter()
             .zip(self.walks.iter())
             .map(|(lever,walk)| lever + walk.sample(&mut rand::thread_rng()))
             .collect();
    }
  }

  impl Bandit for BanditNonStationary {

    fn use_lever(&mut self, lever: usize) -> f64 {
      let result = Normal::new(self.levers[lever],self.std)
               .unwrap()
               .sample(&mut rand::thread_rng());
      self.update();
      result

    }

  }
}
