mod policies {

  enum Action {
    Explore,
    Lever(usize),
  }

  trait Policy {
    // Choose the action: either a lever for exploiting
    // or the exploring option.
    fn decide(&self) -> Action;

    // Update its values based on the result of the
    // step.
    fn update(&mut self, lever : usize, reward : f64);
  }

  trait Estimator {
    // Give the current estimate of the required lever.
    fn estimate(&self, lever : usize) -> f64;

    // Update the estimate of the lever according to
    // the reward.
    fn update(&mut self, lever : usize, reward : f64);
  }

  struct SampleAverage {
    counter : f64,
    estimates : Vec<f64>,
  }

  impl SampleAverage {

    fn new(nb_levers : usize) -> Self {
      SampleAverage {
        counter : 1.0,
        estimates : vec![0.0;nb_levers],
      }
    }
  }

  impl Estimator for SampleAverage {

    fn estimate(&self, lever : usize) -> f64 {
      self.estimates[lever]
    }

    fn update(&mut self, lever : usize, reward : f64) {
      self.estimates[lever] =
        self.estimates[lever] + (reward - self.estimates[lever])/self.counter;
      self.counter += 1.0;
    }
  }


}
