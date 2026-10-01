import { Component, type ErrorInfo, type ReactNode } from "react";

interface State {
  error: Error | null;
}

/** Evita tela em branco quando um componente de resultado recebe dados inesperados. */
export class ErrorBoundary extends Component<{ children: ReactNode; resetKey?: string }, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidUpdate(previous: { resetKey?: string }) {
    if (this.state.error && previous.resetKey !== this.props.resetKey) this.setState({ error: null });
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Falha ao renderizar:", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="alert" role="alert">
          <strong>Algo deu errado ao exibir esta página.</strong>
          <p>{this.state.error.message}</p>
          <button onClick={() => this.setState({ error: null })}>Tentar novamente</button>
        </div>
      );
    }
    return this.props.children;
  }
}
