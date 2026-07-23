# Narração do Vídeo — Fase 3 (minuto a minuto)

**Como usar:** cada bloco tem **[TELA]** (o que mostrar/fazer) e **[FALA]** (o
texto para ler em voz alta). Leia a FALA no ritmo natural; enquanto um comando
roda, continue narrando. Duração-alvo: ~14 min (limite de 15).

**Antes de gravar:**
- Abra um terminal na pasta do projeto e rode uma vez: `export LLM_BACKEND=mock`
  (garante que tudo funcione offline, ao vivo).
- Deixe abertos, em abas: o notebook `notebooks/08_finetuning_langchain.ipynb`,
  e as imagens `results/figures/finetuning_loss.png` e `results/figures/fluxo_langchain.png`.
- Tenha à mão o arquivo `results/finetuning/audit_events.jsonl`.

---

## 0:00 – 1:00 — Abertura

**[TELA]** Slide de título com o nome do grupo e "Tech Challenge – Fase 3". Depois, mostre a pasta do projeto (a árvore de arquivos em `src/assistant/`).

**[FALA]**
> "Olá! Somos o grupo [nomes] e este é o Tech Challenge da Fase 3 da pós de Machine Learning Engineering.
> Nas fases anteriores, a gente construiu, para um hospital, um sistema que previa o desfecho clínico de pacientes com síndrome respiratória aguda grave, a SRAG, e depois otimizou esses modelos.
> Agora, na Fase 3, demos o próximo passo: criamos um **assistente virtual médico**. Ele responde dúvidas clínicas com base nos protocolos internos do hospital, consulta a ficha dos pacientes e coordena um fluxo de decisão automático e seguro.
> Vou mostrar tudo funcionando, do treinamento da IA até o assistente respondendo e os mecanismos de segurança."

---

## 1:00 – 4:30 — Parte 1: Dados e treinamento da IA (fine-tuning)

**[TELA]** No terminal, rode:
```
python scripts/gen_synthetic_data.py
```
Mostre a pasta `data/knowledge_base/` que foi criada (protocolos, FAQ, laudos, prontuários).

**[FALA]**
> "Toda IA precisa aprender com dados. Uma IA genérica não conhece as regras deste hospital — então a gente precisa 'ensinar' ela com os materiais internos.
> Aqui eu gero essa base: protocolos clínicos, as perguntas mais frequentes dos médicos, modelos de laudo e as fichas dos pacientes.
> Um detalhe importante: **todos esses dados são sintéticos e anonimizados** — nenhuma informação real de paciente. Isso respeita a privacidade e a LGPD."

**[TELA]** Abra o notebook, rode a célula que mostra a anonimização (`data_prep.anonimizar`), com o exemplo do CPF e e-mail sendo mascarados.

**[FALA]**
> "Antes de treinar, a gente limpa os dados e apaga qualquer dado pessoal. Olhem: aqui eu passo um texto com nome, CPF e e-mail, e o sistema substitui automaticamente por marcadores. É a etapa de anonimização e curadoria que o enunciado pede."

**[TELA]** No notebook, rode a célula `carregar_externos()`, mostrando uma amostra do PubMedQA e do MedQuAD. Cite o arquivo `data/knowledge_base/external/CITATIONS.md`.

**[FALA]**
> "Além dos dados internos do hospital, a gente também incorporou os dois datasets que o próprio enunciado sugere: o **PubMedQA**, com perguntas de pesquisa médica, e o **MedQuAD**, com perguntas de saúde de sites oficiais do NIH.
> Fizemos uma abordagem híbrida: usamos os dados do hospital, que é o que o enunciado exige, e somamos esses dois datasets como conhecimento complementar. As licenças e as citações estão documentadas no projeto, como pede o uso acadêmico desses dados."

**[TELA]** No terminal:
```
python -m src.assistant.finetuning.dataset_builder
```
Mostre a saída com a composição: 627 exemplos no total.

**[FALA]**
> "Com tudo limpo, montamos o conjunto de treinamento no formato pergunta-e-resposta. No total são 627 exemplos: os dados do hospital em português, mais o PubMedQA e o MedQuAD em inglês. Repare na composição que aparece na tela."

**[TELA]** No terminal:
```
python -m src.assistant.finetuning.train --mode demo
```
Depois abra a imagem `results/figures/finetuning_loss.png`.

**[FALA]**
> "Agora o treinamento em si, o fine-tuning. Como estou numa máquina comum, sem placa de vídeo potente, rodo em **modo demonstração**, que executa todo o processo e gera este gráfico.
> Essa linha caindo é a curva de perda: ela mostra a IA 'errando cada vez menos' a cada rodada — ou seja, aprendendo. Ela começa em torno de 1,1 e cai para cerca de 0,25.
> E deixo claro: o código do treinamento **real**, com a técnica LoRA, está pronto no projeto; basta rodar numa máquina com GPU trocando um parâmetro."

---

## 4:30 – 8:00 — Parte 2: O assistente respondendo (LangChain)

**[TELA]** No notebook, rode a célula da pergunta: *"Qual o alvo de saturação em oxigenoterapia?"*. Destaque na resposta o nome do protocolo (PROT-SRAG-02) e o aviso de validação.

**[FALA]**
> "Feito o treinamento, vamos ao assistente. Quando um médico faz uma pergunta, ele procura o trecho certo nos protocolos, e responde com base nisso.
> Reparem em duas coisas na resposta: primeiro, ele **cita a fonte** — de qual protocolo tirou a informação. Isso é a explicabilidade: o médico pode conferir, não é a IA chutando. Segundo, vem sempre um **aviso de que a conduta precisa de validação médica**."

**[TELA]** Rode a célula do guardrail: *"Prescreva a dose exata de corticoide"*. Mostre que foi bloqueado.

**[FALA]**
> "E aqui uma trava de segurança essencial. Se alguém pedir uma prescrição ou uma dose direta, o assistente **recusa** — porque prescrever é ato médico. Ele não inventa dose; ele redireciona para o protocolo."

**[TELA]** Rode a célula que mostra o resumo de um paciente (ex.: `PAC-0003`) e a resposta contextualizada, com os exames pendentes.

**[FALA]**
> "O assistente também consulta a ficha do paciente. Aqui eu escolho um paciente da base: o sistema puxa os dados dele — saturação, exames pendentes — e responde considerando esse contexto específico. É a resposta personalizada que o enunciado pedia."

**[TELA]** Rode a célula da pergunta geral em inglês: *"What are the symptoms of leukemia?"*. Destaque a fonte `MedQuAD:CancerGov` na resposta.

**[FALA]**
> "E não fica só nos protocolos do hospital. Se a pergunta for mais geral, o assistente busca no MedQuAD, aquele dataset que integramos. Olhem: ele responde e cita a fonte, o MedQuAD. Ou seja, temos uma base de conhecimento híbrida: os protocolos internos em português para SRAG, e o MedQuAD como referência médica mais ampla — sempre indicando de onde veio a resposta."

---

## 8:00 – 11:00 — Parte 3: O fluxo de decisão automático (LangGraph)

**[TELA]** No terminal:
```
python -m src.assistant.cli fluxo --paciente PAC-0001
```
Aponte para a "Trilha do fluxo" e o "ALERTA" no resultado.

**[FALA]**
> "Agora a parte que coordena tudo sozinha: o fluxo de decisão automático.
> Eu entrego um paciente ao sistema e ele executa as etapas em sequência: verifica o nível de risco, checa quais exames estão pendentes, e então decide o que fazer.
> Este paciente é de risco alto, o vermelho. Então o sistema automaticamente **dispara um alerta para a equipe médica** e ainda aponta o exame que falta. Vejam aqui embaixo a 'trilha': triagem, verificar exames, emitir alerta, consolidar. Ele percorreu esse caminho sozinho."

**[TELA]** Rode o fluxo para um paciente verde ou amarelo (ex.: `python -m src.assistant.cli fluxo --paciente PAC-0004`). Mostre que a trilha passa por "sugerir_conduta" em vez de alerta.

**[FALA]**
> "Se eu rodar para um paciente de risco mais baixo, o caminho é diferente: em vez de alerta, ele segue para **sugerir a conduta** baseada no protocolo. Ou seja, o fluxo se adapta ao risco do paciente."

**[TELA]** Abra a imagem `results/figures/fluxo_langchain.png`.

**[FALA]**
> "Este diagrama resume a arquitetura: à esquerda, o assistente que responde e cita a fonte; à direita, o fluxo automático que decide entre alertar ou sugerir conduta. E tudo é registrado para auditoria."

---

## 11:00 – 13:00 — Parte 4: Segurança, auditoria e explicabilidade

**[TELA]** Abra o arquivo `results/finetuning/audit_events.jsonl`. Mostre um registro com pergunta, fontes, backend e a decisão.

**[FALA]**
> "Falando em auditoria: toda interação do assistente fica registrada, como uma caixa-preta. Cada registro guarda a pergunta, o paciente, as fontes que ele usou e as decisões de segurança.
> Isso permite, depois, reconstruir exatamente por que o assistente respondeu aquilo — fundamental num sistema de saúde."

**[FALA]** (recapitulando, sem precisar de tela nova)
> "Então, resumindo os três pilares de segurança: o assistente **nunca prescreve** sem validação humana; toda resposta traz o **aviso de validação**; e tudo é **registrado e rastreável**. Além disso, ele sempre **cita a fonte**, o que dá transparência."

---

## 13:00 – 15:00 — Parte 5: Avaliação e encerramento

**[TELA]** No terminal:
```
python -m src.assistant.finetuning.evaluate
```
Mostre as métricas na tela.

**[FALA]**
> "Por fim, avaliamos o assistente sobre um conjunto de perguntas de referência. Os números: ele encontrou o protocolo correto em **100% das vezes**; nas perguntas gerais, recuperou a fonte certa no MedQuAD em **100%** dos casos; colocou o aviso de validação em **100%** das respostas; e bloqueou pedidos de prescrição em **100%**.
> A cobertura de conteúdo ficou em torno de 60% — o que é esperado nesta demonstração offline e tende a subir bastante com a IA treinada de verdade. Na tela também aparece a composição do dataset: os dados do hospital mais o PubMedQA e o MedQuAD."

**[TELA]** No terminal:
```
pytest tests/test_finetuning.py tests/test_assistant.py -v
```
Mostre "18 passed".

**[FALA]**
> "E para garantir que tudo funciona, temos 18 testes automatizados cobrindo o pipeline — todos passando."

**[TELA]** Slide de encerramento.

**[FALA]**
> "Para fechar: a gente entregou um assistente médico completo — treinado com dados internos anonimizados e com os datasets sugeridos, PubMedQA e MedQuAD; com respostas fundamentadas que citam a fonte, consulta às fichas dos pacientes, um fluxo de decisão automático e seguro, e mecanismos de auditoria.
> A gente atendeu tanto o requisito de usar os dados do hospital quanto a sugestão de datasets. E a arquitetura é flexível: trocando só um parâmetro, o mesmo sistema passa a rodar com a IA treinada de verdade, sem mexer no resto do código.
> Obrigado!"

---

### Dicas rápidas de gravação
- Se um comando demorar, continue narrando a explicação — não deixe silêncio.
- Aumente a fonte do terminal antes de gravar (legibilidade).
- Se algo falhar ao vivo, tenha um print/resultado salvo como reserva.
- Cronometre um ensaio: se passar de 15 min, corte a recapitulação da Parte 4.
