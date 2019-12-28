extern crate rand;
extern crate rand_distr;

use rand::Rng;
use rand_distr::{Normal, Distribution};

trait Bandit {
    // Get reward from a lever.
    fn use_lever(&mut self, lever : usize) -> f64;

    // Choose a lever uniformly at random for exploration.
    // Returns the pair of the lever and its reward.
    fn explore(&mut self) -> (usize,f64);
}

struct BanditStationary {
    nb_levers : usize,
    levers : Vec<Normal<f64>>,
}

impl BanditStationary {

    fn new(inits : Vec<(f64,f64)>) -> Self {
        let levers = inits.iter()
                          .map(|(mean,std)| Normal::new(*mean,*std).unwrap())
                          .collect();
        BanditStationary {
            nb_levers : inits.len(),
            levers
        }
    }
}

impl Bandit for BanditStationary {

    fn use_lever(&mut self, lever: usize) -> f64 {
        self.levers[lever].sample(&mut rand::thread_rng())
    }

    fn explore(&mut self) -> (usize,f64) {
        let lever = rand::thread_rng().gen_range(0,self.nb_levers);
        (lever,self.use_lever(lever))
    }
}

struct BanditNonStationary {
    nb_levers : usize,
    levers : Vec<f64>,
    std : f64,
    walks : Vec<Normal<f64>>,
}

impl BanditNonStationary {

    fn new(nb_levers : usize, init : (f64,f64), walk : (f64,f64)) -> Self {
        BanditNonStationary {
            nb_levers,
            levers : vec![init.0; nb_levers],
            std : init.1,
            walks : (1..nb_levers).map(|_| Normal::new(walk.0,walk.1).unwrap())
                                  .collect(),
        }
    }

    fn update(&mut self) {
        self.levers =
            self.levers.iter()
                       .zip(self.walks.iter())
                       .map(|(lever,walk)| lever + walk.sample(&mut rand::thread_rng()))
                       .collect();
    }
}

impl Bandit for BanditNonStationary {

    fn use_lever(&mut self, lever: usize) -> f64 {
        let result = Normal::new(self.levers[lever],self.std)
                           .unwrap()
                           .sample(&mut rand::thread_rng());
        self.update();
        result

    }

    fn explore(&mut self) -> (usize,f64) {
        let lever = rand::thread_rng().gen_range(0,self.nb_levers);
        let result = (lever,self.use_lever(lever));
        self.update();
        result
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
