import styled from 'styled-components';

const SidebarContainer = styled.div`
  width: 320px;
  border-right: 1px solid ${props => props.theme.colors.lightGray};
  height: 100%;
  display: flex;
  flex-direction: column;
`;

const SidebarHeader = styled.div`
  padding: 16px;
  border-bottom: 1px solid ${props => props.theme.colors.lightGray};
  font-weight: 500;
  font-size: 18px;
`;

const SidebarContent = styled.div`
  flex: 1;
  overflow-y: auto;
`;

const ChatSidebar = ({ children }) => {
  return (
    <SidebarContainer>
      <SidebarHeader>Chats</SidebarHeader>
      <SidebarContent>
        {children}
      </SidebarContent>
    </SidebarContainer>
  );
};

export default ChatSidebar;