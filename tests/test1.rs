use bandit_rs;
use bandit_rs::{policies,problems};
use policies::estimators;

const NB_LEVERS:usize = 10;
const NB_TRIES:usize = 2000;
const LEN_EXP:usize = 1000;
const GAUSS:(f64,f64) = (0.0,1.0);
const EPS:f64 = 0.1;
const NAME: &str = "e = 0.1, sample average";
const EPS2:f64 = 0.01;
const NAME2: &str = "e = 0.01, sample average";
const EPS3:f64 = 0.0;
const NAME3: &str = "e = 0, sample average";

#[test]
fn experiment() {

  let problems : Vec<problems::BanditStationary> =
    (0..NB_TRIES).map( |_| problems::BanditStationary::new(NB_LEVERS,GAUSS,&mut rand::thread_rng()))
                 .collect();
  let est = estimators::SampleAverage::new(NB_LEVERS);

  let mut policies = Vec::new();
  policies.push(policies::EGreedy::new(NB_LEVERS,EPS,est.clone()));
  policies.push(policies::EGreedy::new(NB_LEVERS,EPS2,est.clone()));
  policies.push(policies::EGreedy::new(NB_LEVERS,EPS3,est.clone()));

  let results : Vec<Vec<f64>> =
    bandit_rs::optimal_percentage(bandit_rs::run_experiments(policies,problems,NB_TRIES,LEN_EXP),
                                  NB_TRIES,
                                  LEN_EXP);

  let mut names = Vec::new();
  names.push(NAME);
  names.push(NAME2);
  names.push(NAME3);

  bandit_rs::plot_results(names.into_iter().zip(results.into_iter()).collect(), LEN_EXP);
}
