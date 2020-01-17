use bandit_rs::{BanditInit,EstimatorInit,PolicyInit};

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

  let problem : BanditInit =
    BanditInit::NonStationaryInit {nb_levers : NB_LEVERS,
                                   init_vals : GAUSS,
                                   walk : WALK};

  let est1 = EstimatorInit::SampleAverageInit {nb_levers : NB_LEVERS};
  let est2 = EstimatorInit::ConstantStepInit {nb_levers : NB_LEVERS,
                                              step : ALPHA};
  let policies = [ PolicyInit::EGreedyInit {nb_levers : NB_LEVERS,
                                            expl_proba : EPS,
                                            est : &est1},
                   PolicyInit::EGreedyInit {nb_levers : NB_LEVERS,
                                            expl_proba : EPS,
                                            est : &est2}
                 ];

  let results : Vec<Vec<f64>> =
    bandit_rs::optimal_percentage(bandit_rs::run_experiments(&policies,
                                                             problem,
                                                             NB_TRIES,
                                                             LEN_EXP),
                                  NB_TRIES,
                                  LEN_EXP);

  let names = [ NAME, NAME2 ];

  bandit_rs::plot_results(&results,&names, LEN_EXP);
}
