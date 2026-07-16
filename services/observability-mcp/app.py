from fastmcp import FastMCP

mcp = FastMCP("observability")


@mcp.tool
def health() -> str:
    """
    Simple health check.
    """
    return "Observability MCP is running"


if __name__ == "__main__":
    mcp.run()
