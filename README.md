# Value at Risk via GARCH

The following figure shows an example of computing out-of-sample VaR using GARCH models.

![var_plot](gph/var.png)


### Performance Example:

<br>

| PPS | Violation Number | QS | Hit Percentage | 
| :-: | :-: | :-: | :-: |
| -2.641489 | 14 | 0.000489 | 0.015152 |

<br>

* PPS stands for partial predictive score

$$
PPS = - \frac{1}{T_{test}} \sum_{D_{test}} \log p \left(y_t | y_{1:t-1}, \hat{\theta} \right).
$$

* The number of violations is defined as the number of times over the test data $D_{test}$ the observation $y_t$ is outside its 99\% one-step-ahead forecast interval.

* QS stands for quantile score (Taylor, 2019), where $q_{t,\alpha}$ is the $\alpha$-VaR forecast of $y_t$, conditional on $y_{1:t-1}$. The smaller the quantile score, the better the VaR forecast.

$$
QS = \frac{1}{T_{test}} \sum_{D_{test}} (\alpha - I_{y_{t} \leq q_{t,\alpha}}) (y_t - q_{t,\alpha}),
$$

* Hit percentage is defined as the percentage of $y_t$ in the test data that is below its $\alpha$-VaR forecast. The hit percentage is expected to be close to $\alpha$.

### Bibliography

Taylor, J. W. (2019). Forecasting value at risk and expected shortfall using a semiparametric approach based on the asymmetric laplace distribution. _Journal of Business & Economic Statistics_, 37(1):121–133.



