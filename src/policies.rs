mod policies {

  enum Action {
    Explore,
    Lever(usize),
  }

  trait Policy {
    // Choose the action: either a lever for exploiting
    // or the exploring option.
    fn decide() -> Action;

    // Update its values based on the result of the
    // step.
    fn update(lever : usize, reward : f64);
  }

}
