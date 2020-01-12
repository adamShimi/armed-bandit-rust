use bandit_rs;
use bandit_rs::{policies,problems};
use policies::estimators;

const NB_LEVERS:usize = 10;
const NB_TRIES:usize = 2000;
const LEN_EXP:usize = 10000;
const GAUSS:(f64,f64) = (0.0,1.0);
const WALK:(f64,f64) = (0.0,0.10);
const EPS:f64 = 0.1;
const NAME: &str = "e = 0.1, sample average";
const ALPHA:f64 = 0.1;
const NAME2: &str = "alpha = 0.1, constant step";

#[test]
fn experiment() {

  let problems : Vec<problems::BanditNonStationary> =
    (0..NB_TRIES).map( |_| problems::BanditNonStationary::new(NB_LEVERS,GAUSS,WALK))
                 .collect();
  let est1 = estimators::SampleAverage::new(NB_LEVERS);
  let est2 = estimators::ConstantStep::new(NB_LEVERS,ALPHA);

  let mut policies = Vec::new();
  policies.push(policies::EGreedy::new(NB_LEVERS,EPS,Box::new(est1)));
  policies.push(policies::EGreedy::new(NB_LEVERS,EPS,Box::new(est2)));

  let results : Vec<Vec<f64>> =
    bandit_rs::optimal_percentage(bandit_rs::run_experiments(policies,problems,NB_TRIES,LEN_EXP),
                                  NB_TRIES,
                                  LEN_EXP);

  let mut names = Vec::new();
  names.push(NAME);
  names.push(NAME2);

  bandit_rs::plot_results(names.into_iter().zip(results.into_iter()).collect(), LEN_EXP);
}
