from mattermostdriver import Driver
import requests
import os

bot_username = 'drone'
server_url = 'mm.conleylee.com'

def main():
    status = os.environ['DRONE_STAGE_STATUS']
    bot_password = os.environ['MATTERMOST_TOKEN']
    repo = os.environ['DRONE_REPO_NAME']
    branch = os.environ['DRONE_SOURCE_BRANCH']
    repo_link = os.environ['DRONE_REPO_LINK']
    author = os.environ['DRONE_COMMIT_AUTHOR_NAME']
    build_number = os.environ['DRONE_BUILD_NUMBER']
    build_link = os.environ['DRONE_BUILD_LINK']

    if status == 'success':
        message = f'[{repo}/{branch}]({repo_link}/src/branch/{branch}) [build\#{build_number}]({build_link}) {status}. good job!'
    else:
        message = f'[{repo}/{branch}]({repo_link}/src/branch/{branch}) [build\#{build_number}]({build_link}) {status}. follow previous link for more details!'

    bot = Driver({
        'url': server_url,  # no firewall, proxy etc.
        'token': bot_password,
        'port': 443,
        'scheme': 'https',  # no SSL issues
        'verify': False,
    })

    bot.login()
    my_channel_id = bot.channels.get_channel_by_name_and_team_name(
        'stupidcode',
        'Tengine')['id']
    bot.posts.create_post(options={
        'channel_id': my_channel_id,
        'message': message,
    })


if __name__ == '__main__':
    main()
