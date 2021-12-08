# cs107-FinalProject - Group #32
## Members: Xuliang Guo, Kamran Ahmed, Van Anh Le, Hanwen Cui

[![codecov](https://codecov.io/gh/cs107-XKVH/cs107-FinalProject/branch/main/graph/badge.svg?token=SAQEVYPUXC)](https://codecov.io/gh/cs107-XKVH/cs107-FinalProject)

<!-- how to install, basic usage -->

## Broader Impacts and Inclusivity

### Broader Impacts
Virtually all machine learning and AI algorithms can be attributed to solving optimization problems during the training process. While automatic differentiation does not direct broader impacts, its extensive use as an intermediate step in these algorithms forces us to consider the broader impact of our package. First of all, our package will be contributing to biases against African-American and other underrepresented minorities that current ML models used in the criminal justice system or hiring processes are already imposing. Second, any errors in our calculations could lead to misspecified models and erroneous predictions with significant impacts to downstream users. These impacts are especially grave in safety-critical settings such as healthcare, where a model that utilizes a faulty AD library could misdiagnose a patient or suggest sub-optimal treatments. 

### Inclusivity
While our codebase is technically available and open for anyone to contribute through our GitHub repository, there are technical barriers that might prevent certain groups from participating in this process. Any contributors would need to have working knowledge of git version control and principles of software development. This precludes people from rural communities, communities of color, or poor urban communities, who are less likely to receive formal and rigorous training in computer science. Even at the college level, CS curricula are not homogenous and concepts such as git version control might not be taught at every school. Furthermore, users from other disciplines who rely on optimization and AD might be discouraged by the initial fixed cost of learning a complicated system such as git.

Any developer who wants to contribute to our codebase can make a new branch and create a pull request. Pull requests will then be reviewed by one or many members of our team, depending on the extent of contribution. In order to make this process more inclusive, we could include a step-by-step guide on our repository that provides explicit direction on how to work with git and the expected best-practices that we hope they would follow.
