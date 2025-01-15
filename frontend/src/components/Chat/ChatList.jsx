import styled from 'styled-components';

const ListContainer = styled.div`
  border-right: 1px solid ${props => props.theme.colors.lightGray};
  height: 100%;
`;

const ChatItem = styled.div`
  padding: 16px;
  border-bottom: 1px solid ${props => props.theme.colors.lightGray};
  cursor: pointer;
  background: ${props => props.isSelected ? '#f2f7f2' : 'white'};

  &:hover {
    background: #f7faf7;
  }
`;

const UserInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
`;

const Avatar = styled.img`
  width: 40px;
  height: 40px;
  border-radius: 20px;
`;

const UserName = styled.h4`
  margin: 0;
  font-size: 14px;
`;

const LastMessage = styled.p`
  margin: 4px 0 0;
  font-size: 13px;
  color: ${props => props.theme.colors.gray};
`;

const mockChats = [
  {
    id: 1,
    user: {
      name: 'John Doe',
      avatar: 'https://via.placeholder.com/40'
    },
    lastMessage: 'Thanks for the update',
    timestamp: '2h ago'
  },
  // Add more mock chats
];

const ChatList = ({ onSelectChat, selectedChat }) => {
  return (
    <ListContainer>
      {mockChats.map(chat => (
        <ChatItem 
          key={chat.id}
          isSelected={selectedChat?.id === chat.id}
          onClick={() => onSelectChat(chat)}
        >
          <UserInfo>
            <Avatar src={chat.user.avatar} alt={chat.user.name} />
            <div>
              <UserName>{chat.user.name}</UserName>
              <LastMessage>{chat.lastMessage}</LastMessage>
            </div>
          </UserInfo>
        </ChatItem>
      ))}
    </ListContainer>
  );
};

export default ChatList;