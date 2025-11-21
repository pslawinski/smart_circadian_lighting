# Contributing to Smart Circadian Lighting

Thank you for your interest in contributing to Smart Circadian Lighting! This document provides guidelines and information for contributors.

## Development Setup

### Prerequisites
- Python 3.11 or later
- [uv](https://github.com/astral-sh/uv) for dependency management (recommended)
- Git

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/pslawinski/smart_circadian_lighting.git
   cd smart_circadian_lighting
   ```

2. **Set up development environment:**
   ```bash
   # Install development dependencies
   uv pip install -r requirements-dev.txt

   # Or using pip
   pip install -r requirements-dev.txt
   ```

3. **Run tests:**
   ```bash
   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=custom_components/smart_circadian_lighting

   # Run specific test file
   pytest tests/test_state_management.py
   ```

4. **Code quality checks:**
   ```bash
   # Format code
   black .

   # Lint code
   ruff check .

   # Type checking
   mypy custom_components/smart_circadian_lighting
   ```

## Development Workflow

### 1. Choose an Issue
- Check the [GitHub Issues](https://github.com/pslawinski/smart_circadian_lighting/issues) for open tasks
- Comment on the issue to indicate you're working on it
- Create a new branch for your work

### 2. Development Process
```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# Write tests for new functionality
# Ensure all tests pass
pytest

# Format and lint your code
black .
ruff check . --fix

# Commit your changes
git add .
git commit -m "feat: add your feature description"
```

### 3. Testing Requirements
- **All tests must pass** before submitting a PR
- **New features require tests** with good coverage
- **Test both success and failure scenarios**
- **Integration tests** for complex features

### 4. Code Quality Standards
- **Type hints** for all function parameters and return values
- **Docstrings** for all public functions and classes
- **Descriptive variable names** and clear comments
- **Follow existing code patterns** and conventions

### 5. Documentation
- Update README.md for user-facing changes
- Add docstrings for new functions
- Update type hints and comments as needed

## Pull Request Process

### Before Submitting
1. **Rebase** your branch on the latest main branch
2. **Run full test suite** and ensure all tests pass
3. **Run code quality checks** (black, ruff, mypy)
4. **Test manually** if the change affects user experience

### PR Description
- **Clear title** describing the change
- **Detailed description** of what was changed and why
- **Link to related issues** if applicable
- **Screenshots** for UI changes
- **Testing instructions** if needed

### Review Process
- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, your PR will be merged

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors:

- **Be respectful** and considerate in all interactions
- **Be collaborative** - help others when possible
- **Be patient** with new contributors
- **Focus on constructive feedback**

## Areas for Contribution

### High Priority
- **Bug fixes** - especially those affecting user experience
- **Performance improvements** - especially for time-critical operations
- **Documentation improvements** - clearer setup instructions, better examples

### Medium Priority
- **New features** - circadian rhythm algorithms, additional light types
- **UI/UX improvements** - better configuration flow, status displays
- **Testing improvements** - more comprehensive test coverage

### Low Priority
- **Code refactoring** - improving internal structure without changing behavior
- **Additional integrations** - support for more light brands/protocols
- **Internationalization** - translations for additional languages

## Getting Help

- **GitHub Issues**: For bugs, feature requests, and general questions
- **GitHub Discussions**: For longer-form discussions and community support
- **Documentation**: Check the README and docs folder for detailed information

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (Creative Commons Attribution-NonCommercial 4.0 International).

Thank you for contributing to Smart Circadian Lighting! 🎉