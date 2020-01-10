extern crate rand;
extern crate rand_distr;
extern crate rayon;
extern crate gnuplot;

use rayon::prelude::*;
use std::iter::once;
use gnuplot::{Graph, Figure, Caption,AxesCommon,AutoOption};

pub mod problems;
pub mod policies;
pub mod helper;

pub fn run_experiment<T,U>(policies : Vec<U>,
                           problems : Vec<T>,
                           nb_tries : usize,
                           len_exp : usize) -> Vec<Vec<Vec<Step>>>
  where T : problems::Bandit,
        U : policies::Policy {

  let all_exps : Vec<Vec<(U,T)>> =
    policies.into_iter()
            .map(|x| once(x).cycle()
                            .take(nb_tries)
                            .zip(problems.clone()
                                        .into_iter()
                            )
                            .collect::<Vec<(U,T)>>()
            )
            .collect();
  all_exps.into_par_iter()
          .map(|exps|
            exps.into_par_iter()
                .map(|(policy, problem)| {
                  let exp = Experiment::new(problem.clone(),
                                            policy.clone());
                  (0..len_exp).fold(exp,|mut acc,_| {
                                 acc.step();
                                 acc
                              })
                              .get_results()
                })
                .collect::<Vec<Vec<Step>>>()
          )
          .collect()

}

pub fn optimal_percentage(results : Vec<Vec<Vec<Step>>>,
                          nb_tries : usize,
                          len_exp : usize) -> Vec<Vec<f64>> {
  results.iter()
         .map(|exps| exps.iter()
                         .fold(vec![0.0;len_exp],|acc,results|
                            acc.iter()
                               .zip(results.iter())
                               .map(|(acc_val,step)| *acc_val+((step.optimal as usize) as f64))
                               .collect()
                         )
                         .into_iter()
                         .map(|x| x/(nb_tries as f64))
                         .collect()
         )
         .collect()
}

pub fn plot_results(results : Vec<(&str,Vec<f64>)>,
                    len_exp : usize) {

  let mut output = Figure::new();
  let axes =
    output.axes2d()
          .set_title("Average of optimal action in function of time", &[])
          .set_legend(Graph(0.5), Graph(0.9), &[], &[])
          .set_x_label("Time steps", &[])
          .set_y_label("Percentage of optimal actions", &[])
          .set_y_range(AutoOption::Fix(0.0),AutoOption::Fix(1.0));

  let time_steps : &[usize] = &(1..=len_exp).collect::<Vec<usize>>();
  for (name,vals) in results.iter() {
    axes.lines(time_steps,
              &vals[..],
              &[Caption(name)],
    );
  }
  output.show().unwrap();
}

#[derive(Clone)]
pub struct Experiment<T,U>
  where T : problems::Bandit,
        U : policies::Policy {
  problem : T,
  policy : U,
  results : Vec<Step>,
}

impl<T,U> Experiment<T,U>
  where T : problems::Bandit,
        U : policies::Policy {

  pub fn new(problem : T,
             policy : U) -> Self {
    Experiment {
      problem,
      policy,
      results : Vec::new(),
    }
  }

  pub fn get_results(self) -> Vec<Step> {
    self.results
  }

  pub fn step(&mut self) {
    let lever = self.policy.decide();
    let optimal = self.problem.is_optimal(lever);
    let reward = self.problem.use_lever(lever);
    self.policy.update(lever,reward);
    self.results.push(Step { lever, optimal, reward, });
  }

}

#[derive(Clone)]
pub struct Step {
  pub lever : usize,
  pub optimal : bool,
  pub reward : f64,
}

#[cfg(test)]
mod tests {
  #[test]
  fn it_works() {
    assert_eq!(2 + 2, 4);
  }
}
