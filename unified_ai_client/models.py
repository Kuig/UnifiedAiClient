from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

DEFAULT_TIMEOUT = 300
"""Seconds a request may take when neither the call nor the configuration sets one."""

MAX_TOP_LOGPROBS = 20
"""The most alternatives per token ``call_ai(logprobs=...)`` accepts, the common ceiling."""


@dataclass
class ToolDefinition:
    """Definition of a tool that the model can call.

    Pass a list of these to call_ai() via the ``tools`` parameter to expose
    callable functions to the model. The model may then respond with
    ``AiResponse.tool_calls`` instead of (or in addition to) text.

    Attributes:
        name: Unique tool function name. Must be a valid identifier.
        description: Human-readable description of what the tool does and when
            to use it. Clear descriptions improve model accuracy.
        parameters: JSON Schema object describing the function's parameters.
            Must be a valid JSON Schema of type ``"object"`` with a
            ``"properties"`` field.
    """
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    """A tool call requested by the model in an AiResponse.

    When a model decides to invoke a tool, the provider parses the request
    and populates ``AiResponse.tool_calls`` with one or more of these.
    The consumer is responsible for executing the tool and re-invoking
    ``call_ai()`` with a ``ToolResult`` for each call.

    Attributes:
        id: Provider-assigned identifier for this specific tool call. Must be
            passed back in ``ToolResult.call_id`` to correlate the result.
            Providers that do not return an id receive a generated fallback.
        name: The name of the tool function to execute.
        arguments: Parsed argument dictionary for the tool function.
    """
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result of a tool execution, provided by the consumer.

    After receiving ``AiResponse.tool_calls``, the consumer executes each
    tool and wraps the output in a ``ToolResult``. Pass the list of results
    back to ``call_ai()`` via the ``tool_results`` parameter, or append
    ``to_message()`` to the history, to continue the conversation.

    Attributes:
        call_id: The ``id`` of the ``ToolCall`` this result corresponds to.
            Used by providers to correlate result to request.
        name: The name of the tool function that was executed. Required by
            some providers (e.g. Google) for result routing.
        content: The string result of the tool execution.
    """
    call_id: str
    name: str
    content: str

    def to_message(self) -> dict[str, Any]:
        """This result as a history entry, ready to append to ``messages``.

        Returns:
            ``{"role": "tool", "tool_call_id", "name", "content"}``, the shape
            every adapter converts into its own tool-result format.
        """
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "name": self.name,
            "content": self.content,
        }


@dataclass
class TokenLogprob:
    """The log probability of one generated token, as found in ``AiResponse.logprobs``.

    Attributes:
        token: The token's text.
        logprob: Its natural-log probability, at most 0.
        top_logprobs: The most likely tokens at this position, the generated one
            usually among them, most likely first. Empty when the call asked
            for ``logprobs=0``.
    """
    token: str
    logprob: float
    top_logprobs: list[TokenLogprob] = field(default_factory=list)


@dataclass
class AiResponse:
    """Standardized response from any AI provider.

    Attributes:
        text: The generated text response.
        input_tokens: Number of input/prompt tokens consumed.
        output_tokens: Number of output/completion tokens generated.
        reasoning_tokens: Number of tokens used for internal reasoning/thinking.
            Zero if the provider does not report this separately.
        reasoning_text: Full reasoning/thinking text returned by the model.
            Empty string if reasoning was not requested or not supported by the
            provider.
        reasoning_is_summary: True when ``reasoning_text`` is a summary the
            provider generated from the model's actual chain of thought, rather
            than the raw trace. Google returns thought summaries, and so does
            Anthropic from Claude 4 on; Ollama and the OpenAI-compatible
            providers return the raw text.
            Consumers that measure the trace (length, composition, self-
            corrections) must not compare a summarised trace against a raw one.
        tool_calls: List of tool calls requested by the model.
        message: This turn as a history entry, ready to append to ``messages``
            on the next call: ``{"role": "assistant", "content", "tool_calls"}``
            plus a ``provider_state`` holding the provider's own message,
            JSON-serialisable and opaque. The same provider replays that state
            as it is, which is how the signed thinking blocks and thought
            signatures a tool loop requires survive the round trip; any other
            provider ignores it and reads the canonical fields. ``None`` when
            the adapter built no message.
        logprobs: One entry per generated token when the call asked for
            ``logprobs``, ``None`` otherwise. An empty list means they were
            asked for and the provider returned none.
    """
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_text: str = ""
    reasoning_is_summary: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)
    message: dict[str, Any] | None = None
    logprobs: list[TokenLogprob] | None = None


@dataclass(frozen=True)
class ProviderInfo:
    """A provider that can be used right now, as returned by ``list_providers()``.

    Attributes:
        name: Registry name, the value to pass as ``provider=``.
        local: True for a provider that runs on this machine (a local server or
            a subprocess), False for a cloud endpoint.
        requires_api_key: Whether this provider needs a credential. A cloud
            provider listed here has one, or an explicit ``url``.
        file_types: Attachment classes it carries natively (``'image'``,
            ``'audio'``, ``'video'``, ``'document'``). Text files are always accepted and
            never listed. Empty for ``script``, which declares nothing: only the
            script knows what it can open.
        supports_unload: Whether it keeps a model resident and can release it.
        supports_embeddings: Whether ``get_embedding()`` is implemented.
    """
    name: str
    local: bool
    requires_api_key: bool
    file_types: frozenset[str]
    supports_unload: bool
    supports_embeddings: bool


@dataclass(frozen=True)
class ModelInfo:
    """One model a provider offers, as returned by ``list_models()``.

    Every capability field is ``None`` when the provider's model listing does not
    say, which is not the same as ``False``. Most endpoints report little more
    than an id, so ``None`` is the common answer, and a consumer must treat it as
    "unknown" rather than "unsupported".

    Attributes:
        id: Model identifier, the value to pass as ``model=``.
        provider: Registry name of the provider that serves it.
        display_name: Human-readable name, when the provider supplies one.
        reasoning: Whether the model can produce a reasoning trace.
        tool_use: Whether the model supports tool calling.
        file_types: Native attachment classes the model accepts, or ``None``.
        embeddings: Whether this is an embedding-capable model.
        context_length: Context window in tokens.
    """
    id: str
    provider: str
    display_name: str | None = None
    reasoning: bool | None = None
    tool_use: bool | None = None
    file_types: frozenset[str] | None = None
    embeddings: bool | None = None
    context_length: int | None = None


@dataclass
class AiRequest:
    """Internal request object passed to provider adapters.

    Constructed by client.py from call_ai() arguments. Not part of the
    public API: consumers never create this directly.

    Attributes:
        provider: Provider name (e.g., 'ollama', 'google', 'anthropic').
        model: Model identifier or script path (for 'script' provider).
        prompt: User prompt text.
        system_prompt: Optional system instruction.
        messages: Optional chat history as a list of dicts with 'role' and
            'content' keys. Each dict may also contain an optional 'files'
            key with a list of file paths to attach to that message.
        file_path: Optional local file path or list of file paths for
            multimodal input. Accepts images, audio, text files, and PDFs.
            The provider handles all encoding, upload, and fallback internally.
        temperature: Sampling temperature, or ``None`` to fall back to the
            configured value and, failing that, to send nothing and leave the
            provider's own default in place. ``None`` rather than a number for
            the same reason as ``top_k``: with a value there is no way to tell
            "not given" from "given the default", so a temperature set through
            ``configure_provider()`` could never win.
        thinking: Whether to enable extended thinking/reasoning mode (True/False)
            or use the provider's default behavior ("default").
        format_json: Whether to force JSON output format.
        timeout: Maximum seconds to wait for a response. ``None`` on the
            dataclass only; ``call_ai()`` resolves it against the provider's
            configured timeout before any adapter sees the request, so a
            provider always receives a concrete number.
        top_k: Sampling parameter top_k, or ``None`` to send nothing and leave
            the provider's own default in place. ``None`` is the default
            because there is no value that is correct everywhere: OpenAI's
            Chat Completions API rejects the parameter outright.
        top_p: Sampling parameter top_p, or ``None`` for the provider's own
            default.
        max_tokens: Limit on generated tokens, or ``None`` for the provider's
            own default. ``anthropic`` is the exception, because its API
            requires the field: see ``AnthropicProvider.DEFAULT_MAX_TOKENS``.
        sleep_time: Carried for reference only. Rate limiting is applied once
            in ``call_ai()`` before the retry loop, and no provider reads this.
        tools: Optional list of tool definitions the model can use.
        tool_results: Optional list of tool results to include in the prompt.
        extra_options: Optional dict of arbitrary provider-specific options
            merged into the API payload. Keys and values are provider-dependent
            (e.g. {'visual_token_budget': 1120} for Ollama/Gemma4). These
            override any config-level defaults for the same key.
        logprobs: ``None`` for no log probabilities, otherwise how many
            alternatives to return per token, from 0 to ``MAX_TOP_LOGPROBS``.
            Already validated by ``call_ai()``.
    """
    provider: str
    model: str
    prompt: str
    system_prompt: str | None = None
    messages: list[dict] | None = None
    file_path: str | list[str] | None = None
    temperature: float | None = None
    thinking: bool | str = "default"
    format_json: bool = False
    timeout: int | None = None
    top_k: int | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    sleep_time: int | None = None
    tools: list[ToolDefinition] | None = None
    tool_results: list[ToolResult] | None = None
    extra_options: dict | None = None
    logprobs: int | None = None


@dataclass
class EmbeddingRequest:
    """Internal request object passed to a provider's ``embed()``.

    Constructed by client.py from ``get_embedding()`` / ``get_embeddings()``
    arguments, the counterpart of ``AiRequest`` for embeddings. Not part of the
    public API.

    Attributes:
        provider: Provider name.
        model: Embedding model identifier, or script path for 'script'.
        inputs: The texts to embed, one vector per entry. A single entry for
            ``get_embedding()``, which may be empty when only files are given.
        file_path: Local files embedded together with ``inputs[0]`` into one
            vector, or None. Only ever set on a single-input request.
        dimensions: Requested vector size (Matryoshka truncation), or None for
            the model's own. Wins over the same setting in ``extra_options``.
        timeout: Seconds to wait, already resolved against the configuration by
            client.py, so a provider always receives a concrete number.
        extra_options: Call-time provider-specific options, merged over the
            configured ones by the adapter.
    """
    provider: str
    model: str
    inputs: list[str]
    file_path: list[str] | None = None
    dimensions: int | None = None
    timeout: int | None = None
    extra_options: dict | None = None


@dataclass
class ProviderConfig:
    """Provider-specific configuration loaded from config.json.

    This is a flat dataclass holding the configuration for a provider.
    Unrecognized keys from config.json are collected into extra_options.

    Attributes:
        url: Base URL for the provider's API endpoint. ``None`` means "use the
            provider's own ``DEFAULT_URL``", which is how each provider finds
            its standard endpoint without this dataclass having to know them.
            An explicit value is always honoured as given, including one that
            happens to match another provider's default.
        timeout: Maximum seconds to wait for a response.
        sleep_time: Seconds to sleep before each API call. Used for cloud
            provider rate limiting. Defaults to 0 (no sleep).
        extra_options: Dictionary of provider-specific configuration options.
    """
    url: str | None = None
    timeout: int = DEFAULT_TIMEOUT
    sleep_time: int = 0
    extra_options: dict = field(default_factory=dict)
