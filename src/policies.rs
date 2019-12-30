mod policies {

  use rand::Rng;
  use rand::prelude::IteratorRandom;

  enum Action {
    Explore,
    Lever(usize),
  }

  trait Policy {
    // Choose the action: either a lever for exploiting
    // or the exploring option.
    fn decide(&self) -> Action;

    // Update its values based on the result of the
    // step.
    fn update(&mut self, lever : usize, reward : f64);
  }

  pub mod estimators {

    pub trait Estimator {
      // Give the current estimate of the required lever.
      fn estimate(&self, lever : usize) -> f64;

      // Find the list of levers with the best estimate.
      fn optimal(&self) -> Vec<usize>;

      // Update the estimate of the lever according to
      // the reward.
      fn update(&mut self, lever : usize, reward : f64);
    }

    pub struct SampleAverage {
      pub counter : f64,
      pub estimates : Vec<f64>,
    }

    impl SampleAverage {

      pub fn new(nb_levers : usize) -> Self {
        SampleAverage {
          counter : 1.0,
          estimates : vec![0.0;nb_levers],
        }
      }
    }

    impl Estimator for SampleAverage {

      fn estimate(&self, lever : usize) -> f64 {
        self.estimates[lever]
      }

      fn optimal(&self) -> Vec<usize> {
        self.estimates.iter()
                      .enumerate()
                      .fold((self.estimates[0],Vec::new()),
                            |(mut max,mut occs), (nb,est)| {
                              if *est > max {
                                max = *est;
                                occs.clear();
                                occs.push(nb);
                              } else if *est == max {
                                occs.push(nb);
                              }
                              (max,occs)
                      })
                      .1
      }

      fn update(&mut self, lever : usize, reward : f64) {
        self.estimates[lever] =
          self.estimates[lever] + (reward - self.estimates[lever])/self.counter;
        self.counter += 1.0;
      }
    }
  }

  struct EGreedy {
    nb_levers : usize,
    expl_proba : f64,
    estimator : Box<dyn estimators::Estimator>,
  }

  impl EGreedy {

    fn new(nb_levers : usize, expl_proba: f64, estimator : Box<dyn estimators::Estimator>) -> Self {
      EGreedy {
        nb_levers,
        expl_proba,
        estimator,
      }
    }
  }

  impl Policy for EGreedy {

    fn decide(&self) -> Action {
      if rand::thread_rng().gen_bool(self.expl_proba) {
        Action::Explore
      } else {
        Action::Lever(*self.estimator.optimal()
                                     .iter()
                                     .choose(&mut rand::thread_rng())
                                     .unwrap())
      }
    }

    // Update its values based on the result of the
    // step.
    fn update(&mut self, lever : usize, reward : f64) {
      self.estimator.update(lever,reward);
    }
  }

}
