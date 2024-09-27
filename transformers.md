# The principle of operation of transformers in machine learning

---

Существует много разных трансформерных архитектур, и большинство можно разделить на три типа.

## Энкодеры

Модели-энкодеры синтезируют контекстуальные эмбеддинги, которые можно использовать в последующих задачах вроде классификации или распознавания именованных сущностей, поскольку механизм внимания может обрабатывать всю входящую последовательность. Именно этот тип архитектуры мы рассмотрели в этой статье. Самое популярное семейство чистых трансформеров-энкодеров — это BERT и его разновидности.

Передав данные через один или несколько блоков-трансформеров, мы получаем сложную матрицу контекстуализированных эмбеддингов, которая содержит по эмбеддингу на каждый токен последовательности. Но чтобы использовать эти данные для последующих задач вроде классификации нужно сделать одно предсказание. Обычно берут первый токен и передают через классификатор, в котором есть слои Dropout и Linear.

## Декодеры

Этот тип архитектур почти идентичен предыдущему, главное отличие в том, что декодеры используют маски̒рованный (или причинный) слой self-attention, поэтому механизм внимания может принимать только текущий и предыдущие элементы входной последовательности. То есть контекстуальные эмбеддинги учитывают только предыдущий контекст. К популярным моделям-декодерам относится семейство GPT.

## Энкодеры-декодеры

Изначально трансформеры были представлены как архитектура для машинного перевода и использовали и энкодеры, и декодеры. С помощью энкодеров создается промежуточное представление, прежде чем с помощью декодера переводить в желаемый формат. Хотя энкодеры-декодеры сегодня менее распространены, архитектуры вроде T5 показывают, что задачи вроде ответов на вопросы, подведения итогов и классификации можно представить в виде преобразование последовательности в последовательность и решить с помощью описанного подхода.

Главное отличие архитектур типа энкодер-декодер заключается в том, что декодер использует энкодер-декодерное внимание: при вычислении внимания используется результат энкодера (K и V) и входные данные декодера (Q). Сравните с self-attention, когда для всех входных данных используется одна и та же входная матрица эмбеддингов. При этом общий процесс синтеза очень похож на процесс в архитектурах декодеров.


---


There are many different transformer architectures, and most can be divided into three types.

## Encoders

Encoder models synthesize contextual embeddings that can be used in subsequent tasks such as classification or recognition of named entities, since the attention mechanism can process the entire incoming sequence. It is this type of architecture that we have considered in this article. The most popular family of pure transformer encoders is BERT and its varieties.

By transmitting data through one or more transformer blocks, we get a complex matrix of contextualized embeddings, which contains an embedding for each sequence token. But in order to use this data for subsequent tasks like classification, you need to make one prediction. Usually they take the first token and pass it through the classifier, which has Dropout and Linear layers.

## Decoders

This type of architecture is almost identical to the previous one, the main difference is that decoders use a masked (or causal) self-attention layer, so the attention mechanism can only accept the current and previous elements of the input sequence. That is, contextual embeddings take into account only the previous context. Popular decoder models include the GPT family.

## Encoders-decoders

Initially, transformers were presented as an architecture for machine translation and used both encoders and decoders. With the help of encoders, an intermediate representation is created before being translated into the desired format using the decoder. Although decoder encoders are less common today, architectures like T5 show that tasks like answering questions, summarizing, and classifying can be represented as sequence-to-sequence conversion and solved using the described approach.

The main difference between encoder-decoder architectures is that the decoder uses encoder-decoder attention: when calculating attention, the result of the encoder (K and V) and the input data of the decoder (Q) are used. Compare with self-attention, when the same embedding input matrix is used for all input data. At the same time, the overall synthesis process is very similar to the process in decoder architectures.