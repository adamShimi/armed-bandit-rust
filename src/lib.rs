extern crate rand;
extern crate rand_distr;
extern crate dyn_clone;
extern crate rayon;
extern crate gnuplot;

use std::iter::once;

use rand::Rng;
use rayon::prelude::*;
use gnuplot::{Graph, Figure, Caption,AxesCommon,AutoOption};

pub mod experiments;
pub mod problems;
pub mod policies;
pub mod helper;

use experiments::Experiment;
use experiments::Step;

pub fn run_experiments<T,U>(policies : Vec<U>,
                            problems : Vec<T>,
                            nb_tries : usize,
                            len_exp : usize) -> Vec<Vec<Vec<Step>>>
  where T : problems::Bandit,
        U : policies::Policy {

  make_vec_experiment(policies,problems,nb_tries)
    .into_par_iter()
    .map(|exps|
      exps.into_par_iter()
          .map(|exp| exp.steps(len_exp, &mut rand::thread_rng())
          )
          .collect::<Vec<Vec<Step>>>()
    )
    .collect()
}

pub fn run_reprod_experiments<T,U,V> (policies : Vec<U>,
                                    problems : Vec<T>,
                                    rng : &mut V,
                                    nb_tries : usize,
                                    len_exp : usize) -> Vec<Vec<Vec<Step>>>
  where T : problems::Bandit,
        U : policies::Policy,
        V : Rng {

  make_vec_experiment(policies,problems,nb_tries)
    .into_iter()
    .map(|exps|
      exps.into_iter()
          .map(|exp| exp.steps(len_exp, rng))
          .collect::<Vec<Vec<Step>>>()
    )
    .collect()
}

fn make_vec_experiment<T,U>(policies : Vec<U>,
                            problems : Vec<T>,
                            nb_tries : usize) -> Vec<Vec<Experiment<T,U>>>
  where T : problems::Bandit,
        U : policies::Policy {

  policies.into_iter()
          .map(|x| once(x).cycle()
                          .take(nb_tries)
                          .zip(problems.clone()
                                      .into_iter()
                          )
                          .map(|(policy,problem)|
                            Experiment::new(problem.clone(),
                                            policy.clone())
                          )
                          .collect::<Vec<Experiment<T,U>>>()
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

