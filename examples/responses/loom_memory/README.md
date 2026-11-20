# Responses function tools with application-scoped memory

This example adds two explicit preference tools to the Responses API using
Loom's Python client. It demonstrates manual conversation replay and separately
stored preferences. It is an application example, not an OpenAI session store or
an OpenAI-endorsed integration.

The application binds the organization, namespace, agent and user. Tool arguments
contain only a preference key/value or a search query. The write tool returns the
server's receipt unchanged: a `noop` or version zero does not mean a new memory
was written. The read tool preserves provenance, verification, abstention and
degradation metadata. Retrieved content is untrusted data. No prompt or response
is automatically captured as memory.

`respond` sends `store=False`, replays every returned output item in order
(including reasoning items and assistant phases), and pairs each function result
with its original `call_id`. The loop stops after at most four model requests.
The conversation input is held only for that invocation; preferences live in the
configured Loom service according to its permissions and retention policies.

## Optional dependencies and offline checks

Use Python 3.11 or newer. From the repository root, create the normal contributor
environment, then install this example's separate dependency without changing
the SDK lock or consumer dependencies:

```sh
uv sync --locked --all-extras --python 3.11
uv export --locked --all-extras --no-hashes --no-emit-project > .venv/loom-example-constraints.txt
uv pip install --python .venv/bin/python --exclude-newer '8 days' \
  --constraint .venv/loom-example-constraints.txt \
  --requirement examples/responses/loom_memory/requirements.txt
.venv/bin/ty check --python .venv/bin/python --python-version 3.11 examples/responses/loom_memory
.venv/bin/python -m pytest -o addopts= -q examples/responses/loom_memory/test_memory.py
```

The example pins the published `clickhouse-loom==1.1.0` client and uses its direct
`Agent.memory.set` and `Agent.memory.search` methods. No framework adapter or
native session backend is required. Its required isolated CI job installs the
optional client under the repository's unchanged eight-day artifact policy and
runs these checks separately from the SDK's ordinary dependencies.

Tests exercise the actual `AsyncOpenAI` and Loom clients through mocked HTTP
transports. They cover complete output replay, multiple call IDs, fixed identity,
no-op receipts, abstention, provenance, refusals and invalid arguments. They do
not call a model, start a Loom server or require credentials.

## Opt-in live run

This command makes **billable OpenAI requests and real Loom memory requests**.
It requires an existing Loom deployment, a token authorized for the selected
namespace and the deployment's normal write/read protections. A refusal is an
error; the example does not bypass account or session controls. Use a dedicated
example namespace and synthetic preferences. The example does not delete stored
preferences automatically.

Set `OPENAI_API_KEY`, `OPENAI_MODEL`, `LOOM_BASE_URL`, `LOOM_API_KEY`, `LOOM_ORG`,
`LOOM_NAMESPACE` and `LOOM_USER_ID` through your normal secret configuration.
Use named scopes rather than wildcards. Then run:

```sh
.venv/bin/python -m examples.responses.loom_memory.demo --live 'Remember that I prefer an aisle seat.'
.venv/bin/python -m examples.responses.loom_memory.demo --live 'What is my seat preference?'
```

The second invocation supplies no first-invocation conversation history. A
successful recall depends on the configured service and its retrieval verdict;
the example makes no live-storage or recall-quality guarantee.
