use rand::Rng;

use crate::problems::Bandit;
use crate::policies::Policy;


#[derive(Clone)]
pub struct Experiment<T,U>
  where T : Bandit,
        U : Policy {
  problem : T,
  policy : U,
  results : Vec<Step>,
}

impl<T,U> Experiment<T,U>
  where T : Bandit,
        U : Policy {

  pub fn new(problem : T,
             policy : U) -> Self {
    Experiment {
      problem,
      policy,
      results : Vec::new(),
    }
  }

  pub fn steps<V: Rng>(mut self, steps : usize, rng : &mut V) -> Vec<Step> {
    for _ in 0..steps {
      let lever = self.policy.decide(rng);
      let optimal = self.problem.is_optimal(lever);
      let reward = self.problem.use_lever(lever,rng);
      self.policy.update(lever,reward);
      self.results.push(Step { lever, optimal, reward, });
    }
    self.results
  }
}

#[derive(Clone)]
pub struct Step {
  pub lever : usize,
  pub optimal : bool,
  pub reward : f64,
}
