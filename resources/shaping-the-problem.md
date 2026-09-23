Shaping the problem (of transaction foundation models)

The last 18 months have seen a wave of transaction foundation models being announced (Stripe, Mastercard, Revolut, Adyen, Plaid, Visa, Nubank, Sberbank). The concept is straightforward for anyone familiar with machine learning algorithms, and large language models in particular: instead of ingesting and generating language tokens that represent chunks of letters, transaction foundation models ingest and emit records of financial transactions. Once trained, these algorithms can be used to generate predictive or synthetic transactions, but the significant value lies in the representations of the data that they learn along the way, and for this, being fully generative is not always necessary. This is the larger goal of this approach: learning useful representations.

A representation of something can only ever be partial: the famous story of the blind monks who happen upon an elephant, an animal unknown to them, and come to wildly different conclusions about its nature and appearance based on whether they touch its trunk, ear, leg, tail or tusk, has been used to illustrate this point for at least 2500 years. By training foundation models, we are starting out with an advantage over the monks in this parable: we train on all or nearly all data that is available for a particular domain to learn unchangeable, universal laws for that domain (or as close to that as we can get), and do not have to rely on a small, biased dataset of the domain.

However, we are at a disadvantage relative to these monks on a different axis: they can rely on their sense of touch to learn about the new domain, an instrument honed over millions of years of evolution and a lifetime of practice. In training models on novel, non-familiar domains, we do not have any instrument of comparable calibre. Instead, we must reinvent the sensory modalities of the problem domain from scratch, or more prosaically, what data to make available to the model and with what structure. This is what this essay is about, for the particular case of transaction foundation models, but many of these considerations generalise to other modelling domains.

In academic machine learning, many of the problems of interest are already well-posed and embodied in benchmarks. This enables comparability and algorithmic improvements that often generalise across domains. When developing machine learning systems for real-world domains that do not have standardised input data formats or widely understood performance metrics on outputs, a good part of the work shifts from finding the optimal techniques for solving an already-defined optimisation problem to formulating that problem in the first place, so that the solution will be maximally useful in the applied contexts it is intended for. The focus shifts from the algorithmic ‘answer’ to formulating the ‘question’.

We know that the core data primitive of a transaction foundation model is the financial transaction: a record that includes a sender, a receiver, an amount, a denomination, and a timestamp, at a minimum. Beyond this, the field is wide open: what additional data about either the sender or the receiver should be included? Are there other, circumstantial data that could be relevant? Should additional features be precomputed, and should they include relational attributes that encapsulate some of the graph structure either party is embedded in? What are the expected cost-benefit tradeoffs for more complex precomputed features, once their online computation during inference is taken into account? What are their effects on sample efficiency, model size and robustness? Unfortunately, if you are a private company with custom data, no one will be able to answer these questions for you in advance, and experimentation is the only way to narrow down the best available approach.

In particular, we will discuss six technical decisions about the problem representation that must be made before we can even think about the classical modelling decisions, such as model capacity, activation functions, positional embeddings, and so on. We will illustrate a number of the available alternatives using published architectures, although the public literature is far from providing an exhaustive catalogue of implementations.

(A small note on terminology: In language modelling, a “token” is used somewhat ambiguously to refer to a chunk of letters and its corresponding integer id, and often also to the latent embedding this integer value is projected to and fed to the transformer. For some implementations of a transaction foundation model the two meanings can come apart, and we distinguish the two meanings with little indices (1) and (2), where necessary. Where these are omitted, either interpretation is relevant. “token(1)” refers to an integer, while “token(2)” refers to a latent embedding.)

The key technical decisions are the following:

How do we featurise a single transaction? Do we include only the minimal fields, richer additional fields, stateful/historical features about either party, or even graph-based features? Should free-form text be a valid form of input? How are these to be combined?
Do we represent an individual transaction as a single temporal token(2), a fixed number of tokens, or a variable number of tokens?
High-cardinality categorical fields, such as merchant ids, typically need to be condensed in one form or another, to prevent an explosion of trainable parameters and an under-trained model. Which hashing strategy or binning strategy do you want to employ? What level of granularity results in the optimal trade-off, in this context, between model size and training speed on the one hand and precision in representing individual values on the other?
Real-valued input variables, such as the transaction amount, can be represented in multiple ways: they can be discretised and encoded categorically, normalised, log-transformed, and these transformations can also be combined. Which approach is the right one in this case? Is it the same for all real-valued variables?
Do we want to train a fully generative model using a causal objective over transactions, or on individual transaction tokens(1)? Or a BERT-style embedding model using masked reconstruction loss? Or a combination of both?
How do we design the loss function? Do you rescale or weight losses for different target variables? How do you trade them off against each other? Do you need absolute or relative sample weighting?

We will draw on the following papers and reference implementations to illustrate the different approaches taken for each of these steps:

EWE-1, the first open-weights transaction foundation model, trained on transaction records on Ethereum
The NVIDIA reference implementation for a transaction foundation model
The Foundation Purchasing Model (FPM), by Featurespace/Visa, one of the earliest transaction foundation models in the literature
nuFormer, by Nubank, a Brazilian fintech/neobank
TransactionGPT, the more recent contribution by Visa
TREASURE, another Visa transaction foundation model
PRAGMA, the transaction foundation model developed by Revolut
The Sberbank transaction foundation model

It should be noted that there is not yet a comprehensive body of controlled ablations spanning these six design choices. Current models tend to make these decisions based on business context and knowledge, and experiential, intuitive understanding of the researchers. We can therefore only offer considerations and plausible lines of reasoning for how to make them, and not experimentally validated rules.

We will visualise each available choice for a technical decision as a coloured shape. The colour is per technical decision, while the shape is per available choice for that decision. This will enable us towards the end to put these coloured shapes together to get a visual summary of each modelling approach, allowing us to compare the models.

Featurisation

When featurising a transaction, we can combine the following modalities: the raw, minimal transaction record, free text fields associated with the transaction, sender and/or receiver profile fields, engineered global or profile-specific state that incorporates information from prior transactions, and event records other than transactions, such as web or app interactions, or account-internal actions. We will consider each of these in turn.

The minimal transaction record is the atom of the transaction foundation model; its inclusion is pretty much non-negotiable. The real question then is whether this should be the only data available to the model. If the targeted representation is purely behavioural, the available data is massive, and a large majority of accounts have a sufficient number of transactions to draw substantive conclusions from, this is a viable approach. It enables clean, minimal preprocessing and trusts in the scale of the data and the model to learn the interesting relationships that are implicit in the data.

Free text fields, such as payment intent or merchant descriptions, can add an additional layer of semantic granularity that is complementary to the behavioural and explicit field information that might be available to the model otherwise. The semantic space of natural language is enormous, and learning a useful representation of it takes a significant amount of data and compute. The data requirements on the training data balloon, and this can only be partially offset by using an existing, frozen embedding model, as the embedding space itself will add a lot of learnable but spurious statistical relationships to the data. At sufficient scale, these disadvantages can be overcome, and in these circumstances they can enable a more fine-grained semantic understanding of a transaction or an account.

Sender, intermediary or receiver profile fields offer important context to a transaction: they make similarities and differences between counterparties explicit, and thus facilitate the learning of relationships between different discrete values that might appear in a sequence of transactions. For example, merchants A and B might offer the same category of products, and the meaning of having one or the other in the transaction record of a customer might be closely related, but learning that relationship between merchants purely from transactions will take a lot of data and a fairly large model. Encoding dimensions of similarity or difference in explicit variables that are made available to the model makes learning that kind of pattern easier. The main disadvantages are twofold: it adds complexity to the model architecture and the optimisation objective, and it involves significant redundancy if both counterparty fields and ‘first-party’ fields are included and the fields are slow-moving, i.e. stay constant nearly all of the time. More advanced conditioning approaches can mitigate the latter problem, but at the cost of higher implementation complexity.

Engineered state takes a further step: it doesn’t just add data that obtains at the time of the transaction, but rather tracks and aggregates information across transactions and time steps. This can be single-account information, global context or featurised graph-like relationships. This makes a lot of complex information that a model might (or might not) learn from large numbers of minimal transaction records immediately available, which should speed up training and reduce the required level of expressiveness of the model, in addition to potentially adding relevant information that can improve model performance. The main disadvantage is the added complexity in feature engineering, and replicating these transformations on inference. It also relies on assumptions the modeller makes about the problem domain; if these turn out to be wrong, and some of these engineered features are also targets, they might degrade overall performance, rather than improving it through transfer learning.

Other types of events are another source of context for transactions, such as app or web interactions, support interactions, account activity and the like. These are often very heterogeneous both in their internal structure and in the frequency of their occurrence, which makes incorporating them more difficult, and involves many different technical decisions in how to go about it. This additional complexity means that typically it is better to get started without incorporating such data, validating the transaction foundation model architecture on more minimalist data, and checking subsequently whether adding such data improves the performance on downstream tasks. Only if there are strong reasons to believe that such context is essential for learning the data domain should they be incorporated from the beginning.

The published models incorporate these features; unfortunately, not all of them are fully transparent, but they give a good overview of the range of approaches.

EWE-1: counterparty attributes, transaction features, wallet features and time fields. See the full input feature list.

NVIDIA model: amount, merchant name, industry category, merchant category, hour of day, day of week, month, card index, payment method, abridged merchant zip, merchant state, customer id, transaction time delta

Foundation Purchasing Model: Continuous and categorical transaction fields, specifics not disclosed

nuFormer: direction, amount, month, day of month, weekday, transaction description

TransactionGPT: amount, transaction time delta, processing code, merchant id, merchant category, year, month, day of month, weekday, location-related categories, other undisclosed variables

TREASURE: 5 cardholder attributes, 16 transaction attributes, among which are amount, merchant, merchant country, merchant city and merchant category

PRAGMA: direction, amount, currency, description, channel, temporal variables and other undisclosed fields, heterogeneous other data

Sberbank model: amount, transaction category, other undisclosed fields, heterogeneous other data

Tokenization

Once a full featurisation is implemented, the next decision is how to feed that representation to the model. In language models, data is passed to the model as a sequence of integers, or ‘tokens’(1) that represent one or several characters, and the tokenizer that applies this chunking logic is itself learned. In transaction foundation models, the data is more heterogeneous, in that even the minimal record contains three categorical values, the sender, receiver and denomination, and two continuous variables, the amount and the timestamp. Some models choose to adopt the large language modelling approach and transform each of these variables into tokens(1), and pass them to the model one by one, perfectly analogous to how a language model ingests language tokens(1). This requires binning the continuous-valued variables to give them discrete ids. The representation of a single transaction in this approach is ‘sequential’, in that a sequence of tokens represents a transaction, and the model ingests and emits one token at a time.

The main alternative is the ‘parallel’ representation of the different variables associated with a single transaction. Here, the input variables are tokenized and/or projected into a latent space separately, and these latent representations are aggregated into a single latent space through concatenation or per-position averaging or summing, and fed into the model in this form. The model operates on this joint latent space, and the final latent representation within the model is decoded into the target variables, which in an autoregressive context are identical to the input variables. One inference step results in a whole transaction record, rather than a single token(1) that must be followed by many more to represent a full transaction, as is the case with the ‘sequential’ representation.

The disadvantage is that on decoding, conditioning the value of decoded variables on the decoded values of other variables within the same transaction is architecturally more complex, whereas the ‘sequential’ approach achieves this natively. The main advantage is that it radically shortens the sequence length required to represent a transaction history for a given account, by a factor equivalent to the number of variables per transaction. Given the quadratic cost of context size in transformers, this can have significant impacts on training costs at scale. Another advantage is that it allows for the direct representation of, and optimisation over, continuous-valued variables, without any binning and the resulting loss in precision.

Within the ‘sequential’ representation, a further distinction separates fixed-length and variable-length encodings. The latter is especially appealing if some of the variables contain text of variable length — in that case, the model becomes in effect a language model with special additional tokens representing entities and numeric values. The same goes for structured but variable-length input of other types, such as structured JSON; in practice, these fields would be treated as text. When all of the variables that are made available to the model are continuous or discrete, in their original form, the fixed-length sequential representation becomes natural. This allows for more predictable relationships between positions in a sequence of tokens fed to the model, and it typically makes sense to make the model context window a multiple of the number of tokens per transaction.

This is how each of the published transaction foundation models tokenizes their variables:

EWE-1: multivariate transaction record values are embedded separately and concatenated, resulting in one token(2) per transaction

NVIDIA model: each variable in the multivariate transaction records is projected separately into a shared latent space and ordered sequentially, resulting in multiple tokens(2) per transaction

Foundation Purchasing Model: multivariate transaction record values are embedded separately and concatenated, resulting in a single token(2) per transaction

nuFormer: transaction description is tokenized as language, other transaction record values mapped to reserved tokens(1), resulting in multiple and varying tokens(2) per transaction

TransactionGPT: separate, complex embedding logic for heterogeneous variables, starting with separate embeddings that are concatenated, then processed jointly to create joint embeddings, which are then summarised into a smaller number of virtual tokens(2)

TREASURE: per-sequence and per-transaction data are both multivariate, and each is embedded per-variable and concatenated, with the concatenations resulting in an embedding space shared between the two data sources, resulting in one token(2) per transaction, with an additional per-sequence token(2) prepended

PRAGMA: text fields are tokenized as language, other field values are mapped to reserved tokens, and each transaction is aggregated into a single latent representation through a small transformer module, i.e. resulting in one token(2) per transaction

Sberbank model: multivariate event records are projected separately into a latent space, which is transformed through an attention module and the resulting embeddings are pooled into a single embedding, resulting in one token(2) per transaction

High-Cardinality Fields

Most categorical variables in transaction data have a limited number of available values: merchant industry or category, transaction types or settlement currency typically have ranges up to the hundreds, and these can be fed to the transaction foundation model without additional processing. Some fields that track the identity of merchants, products or users, however, have up to millions of possible values, and configuring a transaction foundation model to ingest them as is leads to two significant problems: the number of parameters blows up, as each distinct value gets a distinct learned projection into latent space through the embedding layer, and simultaneously, the long, flat tail of total occurrences of distinct values means that each of these is severely under-trained, and does not capture meaningful information that can be made useful downstream. It is the worst of all possible worlds.

To address this, the cardinality of these fields has to be reduced, i.e. the number of distinct values must be cut by one or more orders of magnitude. This can only be achieved by aggregating sets of original values into a single id, and determining in advance the cardinality of the output id space. There are several approaches to this: hashing the ids leads to random aggregation across ids and deliberate superposition of item semantics onto a single id. Alternatively, the ids can be clustered by some other external measure, and the cluster ids are used instead of the original ones. A third option is to count the occurrence of each value, keep the top N ids identical to themselves, and group the rest into an ‘other’ id. These approaches can also be mixed: top N can be preserved, while the less frequent ids are hashed or clustered, hashing can be applied within clusters, and so on.

A more sophisticated approach is to apply several such mappings in parallel, creating multiple distinct lower-cardinality id spaces for each high-cardinality input variable. Each original value is thereby represented by a combination of lower-cardinality ids. Two values that collide in one mapping typically map to different ids in another, allowing the model to distinguish them from their combined representation. A related compositional approach is to decompose the original value itself into multiple reusable components. For string-valued fields, for example, a language tokenizer can represent a merchant name or other textual identifier as a sequence of subword tokens drawn from a much smaller shared vocabulary. Rather than preserving entity identity directly, this approach preserves information about the internal structure of the value.

This is how each of the published models handles high-cardinality categorical variables:

EWE-1: frequency truncation for high-cardinality fields, 1:1 tokenization for low-cardinality fields

NVIDIA model: hashing or binning for high-cardinality fields, 1:1 tokenization for low-cardinality fields

Foundation Purchasing Model: 1:1 tokenization for disclosed categorical variables, no aggregation strategy mentioned

nuFormer: language tokenization for text combined with 1:1 tokenization for other variables into mutually exclusive subspaces of a single token space

TransactionGPT: compositional hashing for high-cardinality fields, 1:1 tokenization for low-cardinality fields

TREASURE: undisclosed aggregation for high-cardinality fields, 1:1 tokenization for low-cardinality fields

PRAGMA: language-based tokenization for high-cardinality fields, 1:1 tokenization for low-cardinality fields

Sberbank model: 1:1 tokenization, no high-cardinality strategy mentioned

Numeric Variables

The amount of the transaction is one of the most relevant fields in modelling transactions, and there are many more variables that can be useful and are either continuous or at least numeric, such as the total number of transactions associated with a party, the transaction count that day, week, month or year, aggregate measures such as moving averages, the number of counterparties in total or in a given timeframe, and more. All of these variables must be represented in some form to the model, and feeding them in raw wouldn’t do: the variability in scales means that they would interact unpredictably with the model weights, and as target variables would lead to degenerate gradients.

The key decision on how to represent these is closely tied to the approach taken to featurisation and tokenization: if the approach is to adopt an LLM-like architecture consuming a transaction as a sequence of categorical tokens, binning these real-valued variables is your main option. They can be mapped into either distinct or overlapping id spaces: distinct id spaces preserve the variable distinctness in token space but lead to higher token cardinality and larger embedding modules, while overlapping id spaces reduce token cardinality but rely on the model to effectively learn to distinguish the different positional semantics of each bin id.

The alternative, multivariate featurisation and tokenization approach gives more flexibility: the continuous or numeric variables can be binned and passed to the model as categorical values, normalised and passed to the model, embedded via a per-variable feed-forward layer or embedded together with other such variables in more complex forms, such as convolution or attention.

In either case, it often makes sense to log-transform continuous or integer variables, because their empirical distributions are frequently long-tailed. Especially when binning such variables, a fairly even distribution over bins is desirable, and any reversible transformation that achieves such a distribution should be considered. If the model consumes the continuous variable directly, such evenness is slightly less essential, but still very much beneficial. With this approach, and whatever transformations come before, z-normalisation at the very end is strongly recommended, as many of the best practices and algorithmic approaches in deep learning rely on input variables being centered around 0 and having limited variance.

This is how the published models approach continuous variables:
EWE-1: continuous variables are normalised and projected through per-variable linear layers

NVIDIA model: maps ‘amount’ into predefined bins

Foundation Purchasing Model: continuous, normalised representation of continuous variables, no log- or other transformations mentioned

nuFormer: ‘amount’ is mapped to 21 separate bins, with transaction sign recorded separately, no other continuous variables are included

TransactionGPT: ‘amount’ is log-transformed before being embedded

TREASURE: continuous variables are log-transformed, concatenated and projected into the embedding space

PRAGMA: continuous variables are mapped to percentile-derived bins and treated as categorical downstream

Sberbank model: continuous variables are normalised and projected into the embedding space

Optimisation Target

Once we have fully specified what data the model sees, and how, we need to decide what it should do with it. The two major self-supervised objectives used in large sequence models are masked reconstruction and prediction. The difference between these is the temporal relationship between the data that is visible to the model and the target: in masked reconstruction, the model can use data points both prior to and after the obscured target data, while in prediction only prior data points are available. Typically, the task in prediction is the next target value, rather than larger offsets or multiple values, and this is referred to as a ‘causal’ optimisation target. These two optimisation target families correspond to the BERT family (masked reconstruction) and GPT family (causal) in natural language processing.

The specifics of the implementation of either approach depend on the featurisation and tokenization. In using masked reconstruction loss, the choice is over the proportion and distributions of tokens(2) to mask out, and if each transaction corresponds to multiple tokens(2), whether these are excluded or included in the mask as a block. It can also be desirable to mask out tokens(2) representing several adjacent transactions to learn longer-horizon representations, and these blocks of transactions for masking can be sampled from various probability distributions.

With a causal/autoregressive objective, the design decisions on the optimisation target are more limited, and flow more directly from the tokenization approach. If each transaction is a token(2), this straightforwardly translates into predicting the next transaction from the prior history of transactions. If each transaction is represented by a fixed or variable number of tokens(2), the model learns to predict the next fractional piece of information about a transaction from prior transactions and prior information about the same transaction. On inference, this allows for conditional dependencies between fields of the same transaction, but it does make it trickier to extract a single representation per transaction, as no single activation of the model is optimised to capture all the relevant information about a single transaction equally.

These are the optimisation targets used for each of the published foundation models:
EWE-1: causal next-transaction objective

NVIDIA model: causal next-token(1) objective

Foundation Purchasing Model: causal next-transaction objective paired with an auxiliary reconstruction objective, which is moderated through a temporal distance decay term

nuFormer: causal next-token(1) objective

TransactionGPT: causal next-transaction objective

TREASURE: causal next-event objective

PRAGMA: masked reconstruction objective with masks at per-token(2), per-transaction (multiple tokens(2)) and multiple-transaction levels

Sberbank model: causal next-event objective

Loss Composition

The last design decision to be made is what the exact error signal should be. In self-supervised learning, typically all the input variables are also target variables. This isn’t mandatory, however: some variables might provide useful signal about the targets of interest, even if their inclusion in the set of targets would degrade the usefulness of the learned embedding. The opposite can also be the case: labels that cannot be derived directly from the data, but are available for historical data, might shape the learned embeddings in useful ways as targets. Since they won’t be available on inference, they should not be included as input variables. If such auxiliary target variables are included, we move from a fully self-supervised training paradigm to a more hybrid setup, partly self-supervised and partly supervised. In summary, the available variables fall into four buckets: those used as input and target variables, those used as only input or only target variables, but not both, and those excluded from the model completely.

Once the input and target variables are set, the next set of questions crops up: what loss functions should we use, and how should they be aggregated? For categorical variables, the conventional choice is cross entropy loss, but variants and auxiliary loss functions can shape the learning process in various more specific ways. For real variables, the most common loss functions are mean absolute error and mean squared error, but many alternatives from the classical statistics and machine learning literature are available. When training models at scale, the most common losses are also the safest, as the body of experiential knowledge of training dynamics typically is based on them and the numerical implementations are empirically sound.

In the single-variable case, where that variable serves as both input and target, the question of loss aggregation doesn’t apply. What remains is the question of whether samples should be weighted differently, but in foundation model training, this is also typically answered in the negative.

When multiple variables serve as targets, the question of their aggregation becomes more complex: different targets might contain varying amounts of information about the domain, and if different loss functions are used for different target variables, the relative scale intrinsic to each of them needs to be accounted for. Additionally, the spill-over or ‘within-model transfer learning’ effect can vary depending on the relative prominence of the losses of various target variables. How they are aggregated shapes the loss landscape fundamentally, and might interact with the learning rate, learning rate scheduler and optimizer dynamics. This can seem daunting, but in many cases many variants of the same problem are quite well-behaved. It often suffices to start with a configuration that makes intuitive sense, and iterate from there.

These are the loss functions and aggregations applied by the published foundation models:
EWE-1: cross entropy loss for categorical variables, MAE or MSE for continuous variables, with a custom weighted sum used for loss aggregation

NVIDIA model: cross entropy for all variables

Foundation Purchasing Model: MSE for numerical variables and cross entropy for categorical variables; feature losses are summed, past-reconstruction losses are temporally weighted, and the next-prediction and reconstruction objectives are combined with a tunable weight.

nuFormer: cross entropy loss for all variables

TransactionGPT: MSE on log-transformed continuous variables, cross entropy for categorical variables, aggregated via weighted sum, with exact weights not disclosed

TREASURE: Gaussian negative log-likelihood loss for continuous variables, cross entropy for low-cardinality categorical variables and InfoNCE for high-cardinality categorical variables. The individual auxiliary losses are dynamically rescaled with reference to the loss of the abnormal-behaviour target.

PRAGMA: cross entropy loss with smoothing, no loss weighting disclosed

Sberbank model: MSE for numerical variables and cross entropy for categorical variables, no loss weighting disclosed, missing-aware loss

Once these six decisions have been made, the specification of the data preprocessing that is necessary is usually straightforward to derive.  You will likely also have considered some aspects of the model architecture that you want to use as you made these decisions. Once it is fully fleshed out, and you have defined some downstream tasks to evaluate model quality, the normal machine learning workflow begins: iterating on learning rates and learning rate scheduling, optimizers, weight initialization, dropout settings and the like. You are now at the start of a typical academic machine learning problem — congratulations!

Putting the Pieces Together

We can now put together the visual representations of how each foundation model publication combines the decisions in each of the six domains, to give an overview of what shape they give to the learning problem they set themselves.

This allows us to see where individual papers overlap in their approach, and where they diverge. For example, the NVIDIA reference implementation and nuFormer use different featurisation, tokenization and high-cardinality approaches, but the same real-values approach, training objective and loss composition. EWE-1 and TREASURE share their approaches to tokenization, high-cardinality fields and the training objective, but differ in their featurisation.

A small plug

We are the publishers of sequifier, the framework for foundation model development for non-language data. Sequifier is compatible with the majority of approaches and decisions outlined above, and once a model has been defined with the help of its config schema, hyperparameter search and hyperparameter tuning are a fairly straightforward extension. If you want to spend less time on the latter half of the transaction foundation model process, it will help you do that. If you need technical support in applying sequifier to your problem, or want to talk through design decisions on the model or the data, please reach out to leon [at] sistemalabs.com
