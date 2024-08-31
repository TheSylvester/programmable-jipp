# Programmable-Jipp: A Framework for Your Personal, Persistent, Proactive, and Programmable AI Assistant

Programmable-Jipp is a groundbreaking Discord bot framework designed to host and interact with advanced AI assistants. At its core is Jippity, a powerful AI capable of natural language processing, task automation, and proactive engagement. This project pushes the boundaries of human-AI interaction, offering a comprehensive AI management interface and task automation system.

## Table of Contents

- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Command Structure](#command-structure)
- [Contributing](#contributing)
- [Usage Rights](#usage-rights)

## Key Features

1. **Natural Language Programming**: Turn English instructions into executable commands and save them as templates.
2. **Atomic Task Decomposition**: Break down complex tasks into single-responsibility units.
3. **Dynamic Code Generation**: Replace prompts with Python code for improved efficiency.
4. **Multi-LLM Orchestration**: Leverage multiple Language Learning Models for optimal task execution.
5. **Proactive Engagement**: Initiate actions and reach out to users autonomously using cron jobs.
6. **Bot-Swarm Management**: Orchestrate multiple bots as a comprehensive AI management interface.

## System Architecture

- **Programmable-Jipp**: The overarching Discord bot framework, analogous to programmable matter.
- **Jippity**: The core AI intelligence powering the system.
- **Cogs**: Modular components for specific functionalities, accessible via ! commands.

## Installation

```bash
git clone https://github.com/yourusername/programmable-jipp.git
cd programmable-jipp
pip install -r requirements.txt
```

## Usage

1. Set up your Discord bot token in the `.env` file:

   ```
   DISCORD_TOKEN=your_discord_bot_token_here
   ```

2. Run the bot:

   ```bash
   python main.py
   ```

3. Invite the bot to your Discord server and start interacting!

## Command Structure

Programmable-Jipp uses two types of interactions:

1. **@mentions for Jippity**: Interact with the core AI using natural language.

   ```
   @Jippity Make a subroutine for dog breed information
   @Jippity Schedule a daily reminder to drink water at 2 PM
   @Jippity Analyze the sentiment of tweets about AI assistants
   ```

2. **! commands for system operations**:
   - `!list_models`: Display all available models
   - `!model <model>`: Change the currently engaged model
   - `!model`: Show the current model (when used without arguments)
   - `!help`: Display help information
   - `!commands`: List all available commands

Example usage:

```
!list_models
!model gpt-4
!help
```

Note: The @mentions allow for natural conversation with Jippity, enabling it to understand context from the entire message. The ! commands provide quick access to system functions and information.

## Contributing

We welcome contributions to Programmable-Jipp! Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests. Note that all contributions will be subject to the project's usage rights as defined below.

## Usage Rights

All rights reserved. This project is the exclusive property of its creator(s). No part of this software may be reproduced, distributed, or transmitted in any form or by any means, including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the owner(s), except in the case of brief quotations embodied in critical reviews and certain other noncommercial uses permitted by copyright law.

For permission requests or further information, please contact the project owner(s).

---

Programmable-Jipp, with Jippity at its core, represents a new paradigm in human-AI collaboration. It's a framework for creating your personal, persistent, proactive, and infinitely programmable AI companion, ready to transform how you interact with technology and manage your digital world.
