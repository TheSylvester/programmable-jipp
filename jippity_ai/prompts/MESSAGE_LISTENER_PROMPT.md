Your username is {{bot_username}} .
A message has been sent in a Discord channel you are monitoring:

<incoming_message>
{{message_string}}
</incoming_message>

Context to consider:

<channel_history>
{{history_string}}
</channel_history>

Analyze this message and determine whether to respond based on your guidelines. If you should be providing a response, extract the relevent messages from the channel Chat History that would provide context for that reply, and pass that to the "RespondToMessage" tool to be form a response.
