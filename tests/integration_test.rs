use bandit_rs;
use bandit_rs::{policies,problems};
use policies::estimators;

const NB_LEVERS:usize = 10;
const EPS:f64 = 0.1;
const NB_TRIES:usize = 2000;
const LEN_EXP:usize = 2000;
const GAUSS:(f64,f64) = (0.0,1.0);

#[test]
fn experiment() {
  let est = estimators::SampleAverage::new(NB_LEVERS);
  let policy = policies::EGreedy::new(NB_LEVERS,EPS,est);
  let problem = problems::BanditStationary::new(NB_LEVERS,GAUSS);
  let experiment = bandit_rs::Experiment::new(problem,policy);
  let results = bandit_rs::run_experiments(experiment,NB_TRIES,LEN_EXP);
  bandit_rs::plot_optimal_percentage(results,NB_TRIES,LEN_EXP);
}
