ranking-meta-evaluation-data
=====
## Introduction
Rankings are a common system response in a variety of learning tasks like search, recommendation, and NLP.  Developing evaluation metrics for rankings often involves meta-evaluation data consisting of rankings from a variety of systems for a shared set of system inputs.  This repository simplifies gathering meta-evaluation data across several retrieval and recommendation domains. 
### Ranking Data
Given a set of systems, for each request in the domain, we have,
* an incomplete set of item labels, and
* a set of top-k rankings, one per systems.
#### Format
All data are in TREC format with relevance judgments (qrels) in standard four-column trec_eval format,
```
   <query id><subtopic id><document id><relevance grade>
```
where subtopic id is a base 1 identifier (or "0"/"Q0" if there are no subtopic labels); if you are using this outside of the TREC context, you can just use "0" for the second column.  The relevance grade is ordinal with <=0 indicating nonrelevance.  If a query has no documents with relevance grade >0, it is removed from the evaluation.

System runs are assumed to be in standard six-column trec_eval format, with one system per file,
```
   <query id><iteration><document id><rank><score>[<run id>]   
```
where iteration, rank, and run id are often ignored, with the rank  re-computed from the score.  
#### Downloading the Data

**IMPORTANT: You must have a license to access NIST's TREC data.  For more information see [here](https://trec.nist.gov/results.html).**

Set environment variables ``TREC_RESULTS_USER`` and ``TREC_RESULTS_PASSWORD`` and then run ``make``.  Data will be in the ``qrels`` and ``runs`` directories. 

### Statistics
#### Retrieval
| domain tag | requests | systems | rel/request | items/request | reference |
|-----|---------:|-----:|------------:|--------------:|-----------|
| legal/2006 | 39 | 34 | 110.85 | 4835.07 | [paper](https://trec.nist.gov/pubs/trec15/papers/LEGAL06.OVERVIEW.pdf), [www](https://trec.nist.gov/data/legal06.html) |
| legal/2007 | 43 | 68 | 101.023 | 22240.30 | [paper](https://trec.nist.gov/pubs/trec16/papers/LEGAL.OVERVIEW16.pdf), [www](https://trec.nist.gov/data/legal07.html) |
| core/2017 | 50 | 75 | 180.04 | 8853.11 | [paper](https://trec.nist.gov/pubs/trec26/papers/Overview-CC.pdf), [www](https://trec.nist.gov/data/core2017.html) |
| core/2018 | 50 | 72 | 78.96 | 7102.61 | [www](https://trec.nist.gov/data/core2018.html/) |
| deep-docs/2019 | 43 | 38 | 153.42 | 623.77 | [paper](https://trec.nist.gov/pubs/trec28/papers/OVERVIEW.DL.pdf), [www](https://trec.nist.gov/data/deep2019.html) |
| deep-docs/2020 | 45 | 64 | 39.27 | 99.55 | [paper](https://trec.nist.gov/pubs/trec29/papers/OVERVIEW.DL.pdf), [www](https://trec.nist.gov/data/deep2020.html) |
| deep-docs/2021 | 57 | 66 | 189.63 | 98.83 | [paper](https://trec.nist.gov/pubs/trec30/papers/Overview-DL.pdf), [www](https://trec.nist.gov/data/deep2021.html) |
| deep-docs/2022 | 76 | 42 | 1245.62 | 98.86 | [paper](https://trec.nist.gov/pubs/trec31/papers/Overview_deep.pdf), [www](https://trec.nist.gov/data/deep2022.html) |
| deep-docs/2023 | 82 | 5 | 75.10 | 100 | [paper](https://trec.nist.gov/pubs/trec32/papers/Overview_deep.pdf), [www](https://trec.nist.gov/data/deep2023.html) |
| deep-pass/2019 | 43 | 37 | 95.40 | 892.51 | [paper](https://trec.nist.gov/pubs/trec28/papers/OVERVIEW.DL.pdf), [www](https://trec.nist.gov/data/deep2019.html) |
| deep-pass/2020 | 54 | 59 | 66.78 | 978.01 | [paper](https://trec.nist.gov/pubs/trec31/papers/Overview_deep.pdf), [www](https://trec.nist.gov/data/deep2020.html) |
| deep-pass/2021 | 53 | 63 | 191.96 | 99.95 | [paper](https://trec.nist.gov/pubs/trec30/papers/Overview-DL.pdf), [www](https://trec.nist.gov/data/deep2021.html) |
| deep-pass/2022 | 76 | 100 | 628.145 | 97.5 | [paper](https://trec.nist.gov/pubs/trec31/papers/Overview_deep.pdf), [www](https://trec.nist.gov/data/deep2022.html) |
| deep-pass/2023 | 82 | 35 | 49.87 | 99.90 | [paper](https://trec.nist.gov/pubs/trec32/papers/Overview_deep.pdf), [www](https://trec.nist.gov/data/deep2023.html) |
| web/2009 | 50 | 48 | 129.98 | 925.31 | [paper](https://trec.nist.gov/pubs/trec18/papers/WEB09.OVERVIEW.pdf), [www](https://trec.nist.gov/data/web09.html) |
| web/2010 | 48 | 32 | 187.63 | 7013.21 | [paper](https://trec.nist.gov/pubs/trec19/papers/WEB.OVERVIEW.pdf), [www](https://trec.nist.gov/data/web10.html) |
| web/2011 | 50 | 61 | 167.56 | 8325.07 | [paper](https://trec.nist.gov/pubs/trec20/papers/WEB.OVERVIEW.pdf), [www](https://trec.nist.gov/data/web2011.html) |
| web/2012 | 50 | 48 | 187.36 | 6719.53 | [paper](https://trec.nist.gov/pubs/trec21/papers/WEB12.overview.pdf), [www](https://trec.nist.gov/data/web2012.html) |
| web/2013 | 50 | 61 | 182.42 | 7174.38 | [paper](https://trec.nist.gov/pubs/trec22/papers/WEB.OVERVIEW.pdf), [www](https://trec.nist.gov/data/web2013.html) |
| web/2014 | 50 | 30 | 212.58 | 6313.98 | [paper](https://trec.nist.gov/pubs/trec23/papers/overview-web.pdf), [www](https://trec.nist.gov/data/web2014.html) |
| robust/2004 | 249 | 110 | 69.93 | 913.82 | [paper](https://trec.nist.gov/pubs/trec13/papers/ROBUST.OVERVIEW.pdf), [www](https://trec.nist.gov/data/t13_robust.html) |

#### Recommendation
| domain tag | requests | systems | rel/request | items/request | reference |
|-----|---------:|-----:|------------:|--------------:|-----------|
| movielens/2018 | 6005 | 21 | 18.87 | 100.00 | [paper](https://www.dc.fi.udc.es/~dvalcarce/pubs/valcarce-etal-recsys2018.pdf), [www](https://github.com/dvalcarce/evalMetrics) |
| libraryThing/2018 | 7227 | 21 | 13.15 | 100.00 | [paper](https://www.dc.fi.udc.es/~dvalcarce/pubs/valcarce-etal-recsys2018.pdf), [www](https://github.com/dvalcarce/evalMetrics) |
| beerAdvocate/2018 | 17564 | 21 | 13.66 | 99.39 | [paper](https://www.dc.fi.udc.es/~dvalcarce/pubs/valcarce-etal-recsys2018.pdf), [www](https://github.com/dvalcarce/evalMetrics) |
## Contact
[Fernando Diaz](mailto:diazf@acm.org)
