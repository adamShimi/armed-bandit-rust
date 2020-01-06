extern crate rand;
extern crate rand_distr;
extern crate dyn_clone;
extern crate gnuplot;

use gnuplot::{Figure, Caption, Color};

pub mod problems;
pub mod policies;
pub mod helper;

#[derive(Clone)]
pub struct Experiment {
  problem : Box<dyn problems::Bandit>,
  policy : Box<dyn policies::Policy>,
  results : Vec<Step>,
}

impl Experiment {

  pub fn new(problem : Box<dyn problems::Bandit>,
             policy : Box<dyn policies::Policy>) -> Self {
    Experiment {
      problem,
      policy,
      results : Vec::new(),
    }
  }

  pub fn step(&mut self) {
    let lever = self.policy.decide();
    let reward = self.problem.use_lever(lever);
    self.policy.update(lever,reward);
    self.results.push(Step { lever, reward, });
  }


  pub fn optimal_choices(&self) -> Vec<usize> {
    let optimals = self.problem.optimal_levers();
    self.results.iter()
                .enumerate()
                .filter(|(_nb,step)| {
                  optimals.contains(&step.lever)
                })
                .map(|(nb,_step)| nb)
                .collect()
  }
}

#[derive(Clone)]
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
