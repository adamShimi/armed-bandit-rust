extern crate rand;
extern crate rand_distr;

pub mod problems;
pub mod policies;

pub struct Experiment {
  problem : Box<dyn problems::Bandit>,
  policy : Box<dyn policies::Policy>,
}

impl Experiment {

  pub fn new(problem : Box<dyn problems::Bandit>,
         policy : Box<dyn policies::Policy>) -> Self {
    Experiment {
      problem,
      policy,
    }
  }

  pub fn step(&mut self) -> Step {
    let lever = self.policy.decide();
    let reward = self.problem.use_lever(lever);
    self.policy.update(lever,reward);
    Step {
      lever,
      reward,
    }
  }

}

pub struct Step {
  pub lever : usize,
  pub reward : f64,
}

#[cfg(test)]
mod tests {
  #[test]
  fn it_works() {
    assert_eq!(2 + 2, 4);
  }
}
