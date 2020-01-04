pub fn indices_max(vals : &Vec<f64>, to_max: Box<dyn (Fn(&f64) -> f64)>) -> Vec<usize> {
  if vals.is_empty() {
    Vec::new()
  } else {
    vals.iter()
        .map(to_max)
        .enumerate()
        .fold((vals[0],Vec::new()),
              |(mut max,mut occs), (nb,est)| {
                if est > max {
                  max = est;
                  occs.clear();
                  occs.push(nb);
                } else if est == max {
                  occs.push(nb);
                }
                (max,occs)
        })
        .1
  }
}
