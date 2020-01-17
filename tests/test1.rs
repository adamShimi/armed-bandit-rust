use bandit_rs::{BanditInit,EstimatorInit,PolicyInit};

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

  let problem : BanditInit =
    BanditInit::StationaryInit {nb_levers : NB_LEVERS,
                                init_vals : GAUSS};
  let est = EstimatorInit::SampleAverageInit {nb_levers : NB_LEVERS};

  let policies = [ PolicyInit::EGreedyInit {nb_levers : NB_LEVERS,
                                            expl_proba : EPS,
                                            est : &est},
                   PolicyInit::EGreedyInit {nb_levers : NB_LEVERS,
                                            expl_proba : EPS2,
                                            est : &est},
                   PolicyInit::EGreedyInit {nb_levers : NB_LEVERS,
                                            expl_proba : EPS3,
                                            est : &est}
                 ];

  let results : Vec<Vec<f64>> =
    bandit_rs::optimal_percentage(bandit_rs::run_experiments(&policies,problem,NB_TRIES,LEN_EXP),
                                  NB_TRIES,
                                  LEN_EXP);

  let names = [NAME,NAME2,NAME3];

  bandit_rs::plot_results(&results[..], &names, LEN_EXP);
}
