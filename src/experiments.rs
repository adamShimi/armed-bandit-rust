use rand::Rng;

use crate::problems::Bandit;
use crate::problems::BanditEnum;
use crate::policies::Policy;
use crate::policies::PolicyEnum;


#[derive(Clone)]
pub(crate) struct Experiment {
  policy : PolicyEnum,
  problem : BanditEnum,
  results : Vec<Step>,
}

impl Experiment {

  pub(crate) fn new(policy : PolicyEnum,
             problem : BanditEnum) -> Self {
    Experiment {
      problem,
      policy,
      results : Vec::new(),
    }
  }

  pub(crate) fn steps<V: Rng>(mut self, steps : usize, rng : &mut V) -> Vec<Step> {
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
