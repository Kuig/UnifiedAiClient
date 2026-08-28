from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


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
    back to ``call_ai()`` via the ``tool_results`` parameter to continue the
    conversation.

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
            than the raw trace. Google returns thought summaries; Ollama,
            Anthropic and the OpenAI-compatible providers return the raw text.
            Consumers that measure the trace (length, composition, self-
            corrections) must not compare a summarised trace against a raw one.
        tool_calls: List of tool calls requested by the model.
    """
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_text: str = ""
    reasoning_is_summary: bool = False
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class AiRequest:
    """Internal request object passed to provider adapters.

    Constructed by client.py from call_ai() arguments. Not part of the
    public API — consumers never create this directly.

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
        temperature: Sampling temperature.
        thinking: Whether to enable extended thinking/reasoning mode (True/False)
            or use the provider's default behavior ("default").
        format_json: Whether to force JSON output format.
        timeout: Maximum seconds to wait for a response.
        tools: Optional list of tool definitions the model can use.
        tool_results: Optional list of tool results to include in the prompt.
        extra_options: Optional dict of arbitrary provider-specific options
            merged into the API payload. Keys and values are provider-dependent
            (e.g. {'visual_token_budget': 1120} for Ollama/Gemma4). These
            override any config-level defaults for the same key.
    """
    provider: str
    model: str
    prompt: str
    system_prompt: str | None = None
    messages: list[dict] | None = None
    file_path: str | list[str] | None = None
    temperature: float = 0.7
    thinking: bool | str = "default"
    format_json: bool = False
    timeout: int = 300
    top_k: int = 64
    top_p: float = 0.95
    max_tokens: int | None = None
    sleep_time: int | None = None
    tools: list[ToolDefinition] | None = None
    tool_results: list[ToolResult] | None = None
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
    timeout: int = 300
    sleep_time: int = 0
    extra_options: dict = field(default_factory=dict)
