use bandit_rs;
use bandit_rs::{policies,problems};
use policies::estimators;

#[test]
fn experiment() {
  let nb_levers = 10;
  let est = estimators::SampleAverage::new(nb_levers);
  let policy = policies::EGreedy::new(nb_levers,0.1,est);
  let problem = problems::BanditStationary::new(nb_levers,(0.0,1.0));
  let experiment = bandit_rs::Experiment::new(problem,policy);
  bandit_rs::run_experiments(experiment,2000,1000);
}
