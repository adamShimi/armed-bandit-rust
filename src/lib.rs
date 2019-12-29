extern crate rand;
extern crate rand_distr;

mod problems {

    use rand::Rng;
    use rand_distr::{Normal, Distribution};

    trait Bandit {
        // Get reward from a lever.
        fn use_lever(&mut self, lever : usize) -> f64;

        // Choose a lever uniformly at random for exploration.
        // Returns the pair of the lever and its reward.
        fn explore(&mut self) -> (usize,f64);
    }


    // Implementatio of a stationary bandit instance, where
    // nb_levers levers are initialized at according to a normal distribution.
    struct BanditStationary {
        nb_levers : usize,
        levers : Vec<Normal<f64>>,
    }

    impl BanditStationary {

        fn new( nb_levers : usize, init : (f64,f64)) -> Self {
            BanditStationary {
                nb_levers,
                levers : (0..nb_levers).map(|_| Normal::new(init.0,init.1).unwrap())
                                       .collect()
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




    // Implementation of a nonstationary bandit problems, where
    // levers start with the same mean and move according to a random
    // walk at each step.
    struct BanditNonStationary {
        nb_levers : usize,
        levers : Vec<f64>,
        // Only one std, because only the means move according to
        // the random walk; the standard deviation stays the same.
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
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
