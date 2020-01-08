pub fn indices_max(vals : &[f64]) -> Vec<usize> {
  if vals.is_empty() {
    Vec::new()
  } else {
    vals.iter()
        .enumerate()
        .fold((vals[0],Vec::new()),
              |(mut max,mut occs), (nb,est)| {
                if (*est-max) <= std::f64::EPSILON {
                  occs.push(nb);
                } else if *est > max {
                  max = *est;
                  occs.clear();
                  occs.push(nb);
                }
                (max,occs)
        })
        .1
  }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indices_max() {
        assert_eq!(indices_max(&(vec![0.0,1.0])[..]),vec![1]);
    }
}
