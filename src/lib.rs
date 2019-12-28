extern crate rand;
extern crate rand_distr;

use rand_distr::{Normal, Distribution};

trait Bandit {
    // Get reward from a lever.
    fn use_lever(&mut self, lever : usize) -> f64;

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
}

impl Bandit for BanditNonStationary {

    fn use_lever(&mut self, lever: usize) -> f64 {
        let result = Normal::new(self.levers[lever],self.std)
                           .unwrap()
                           .sample(&mut rand::thread_rng());

        self.levers =
            self.levers.iter()
                       .zip(self.walks.iter())
                       .map(|(lever,walk)| lever + walk.sample(&mut rand::thread_rng()))
                       .collect();

        return result

    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
