use bandit_rs::{BanditInit,EstimatorInit,PolicyInit};

const NB_LEVERS:usize = 10;
const LEN_EXP:usize = 10000;
const GAUSS:(f64,f64) = (0.0,1.0);
const EPS:f64 = 0.01;

#[test]
fn experiment() {

  let problem : BanditInit =
    BanditInit::StationaryInit {nb_levers : NB_LEVERS,
                                init_vals : GAUSS};

  let est = EstimatorInit::SampleAverageInit {nb_levers : NB_LEVERS};
  let policy = PolicyInit::EGreedyInit {nb_levers : NB_LEVERS,
                                        expl_proba : EPS,
                                        est : &est};

  bandit_rs::run_parameter_study(&policy,&problem, LEN_EXP, 1..100);
}
