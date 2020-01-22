extern crate rand;
extern crate rand_distr;
extern crate enum_dispatch;
extern crate rayon;
extern crate gnuplot;

use std::iter::once;
use std::ops::Range;

use rand::Rng;
use rayon::prelude::*;
use gnuplot::{Graph, Figure, Caption,AxesCommon,AutoOption};

mod experiments;
mod problems;
mod policies;
mod estimators;
mod helper;

pub use problems::BanditInit;
pub use policies::PolicyInit;
pub use estimators::EstimatorInit;
use problems::create_bandit;
use policies::create_policy;
use experiments::Experiment;
use experiments::Step;

pub fn run_experiments(policies : &[PolicyInit],
                       problem : BanditInit,
                       nb_tries : usize,
                       len_exp : usize) -> Vec<Vec<Vec<Step>>> {

  make_vec_experiment(policies,problem,&mut rand::thread_rng(),nb_tries)
    .into_par_iter()
    .map(|exps|
      exps.into_par_iter()
          .map(|exp| exp.steps(len_exp, &mut rand::thread_rng())
          )
          .collect::<Vec<Vec<Step>>>()
    )
    .collect()
}

pub fn run_reprod_experiments<T> (policies : &[PolicyInit],
                                  problem : BanditInit,
                                  rng : &mut T,
                                  nb_tries : usize,
                                  len_exp : usize) -> Vec<Vec<Vec<Step>>>
  where T : Rng {

  make_vec_experiment(policies,problem,rng,nb_tries)
    .into_iter()
    .map(|exps|
      exps.into_iter()
          .map(|exp| exp.steps(len_exp, rng))
          .collect::<Vec<Vec<Step>>>()
    )
    .collect()
}

fn make_vec_experiment<T>(policies : &[PolicyInit],
                          problem : BanditInit,
                          rng : &mut T,
                          nb_tries : usize) -> Vec<Vec<Experiment>>
  where T : Rng {

  policies.iter()
          .map(|x| once(x).cycle()
                          .take(nb_tries)
                          .map(|policy|
                            Experiment::new(create_policy(policy),
                                            create_bandit(&problem,rng))
                          )
                          .collect::<Vec<Experiment>>()
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

pub fn plot_results(results : &[Vec<f64>],
                    names : &[&str],
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
  for (name,vals) in names.iter().zip(results.iter()) {
    axes.lines(time_steps,
               vals,
               &[Caption(name)],
    );
  }
  output.show().unwrap();
}


pub fn run_parameter_study(policy : &PolicyInit,
                           problem : &BanditInit,
                           len_exp : usize,
                           range : Range<u32>) {

  let policies : Vec<PolicyInit>;
  let step_base : f64;
  let egreedy : bool;
  let ucb : bool;
  match policy {
    &PolicyInit::EGreedyInit {nb_levers, expl_proba, est} => {
      policies =
        range.clone()
             .map(|x| PolicyInit::EGreedyInit {nb_levers,
                                               expl_proba : expl_proba*(2.0_f64).powi(x as i32),
                                               est})
             .collect();
      step_base = expl_proba;
      egreedy = true;
      ucb = false;},
    &PolicyInit::UCBInit {nb_levers, step, est} => {
      policies =
        range.clone()
             .map(|x| PolicyInit::UCBInit {nb_levers,
                                           step : step*(2.0_f64).powi(x as i32),
                                           est})
             .collect();
      step_base = step;
      egreedy = true;
      ucb = false;},

  }
  let results : Vec<f64> =
    policies.iter()
            .map(|policy| {
              let exp = Experiment::new(create_policy(policy),
                                        create_bandit(&problem,&mut rand::thread_rng()));
              let result = exp.steps(len_exp, &mut rand::thread_rng());
              result.into_iter()
                    .map(|x| x.reward)
                    .sum::<f64>() / (len_exp as f64)
            })
            .collect();

  let mut output = Figure::new();
  let axes =
    output.axes2d()
          .set_title("Percentage of optimal action in first 1000 steps", &[])
          .set_legend(Graph(0.5), Graph(0.9), &[], &[])
          .set_x_log(Some(2.0))
          .set_x_label("Value of parameter", &[])
          .set_y_label("Percentage of optimal actions", &[]);

  let time_steps : &[f64] = &range.map(|x| step_base*(2.0_f64).powi(x as i32))
                                  .collect::<Vec<f64>>()[..];
  match (egreedy,ucb) {
    (true,false) => {axes.lines(time_steps,
                               results,
                               &[Caption("EGreedy")],
                              );},
    (false,true) => {axes.lines(time_steps,
                               results,
                               &[Caption("UCB")],
                              );},
    _ => {panic!("Impossible!");},
  }
  output.show().unwrap();
}
