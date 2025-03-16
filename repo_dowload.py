import os
import requests
import zipfile
import io

def obter_branch_padrao(owner: str, repo: str):
    """Obtém o nome do branch padrão do repositório via API do GitHub."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(api_url)

    if response.status_code == 200:
        return response.json().get("default_branch", "main")  # Retorna 'main' se não encontrar
    else:
        print(f"Erro ao obter o branch padrão. Código HTTP: {response.status_code}")
        return None

def baixar_repositorio_github(url_repositorio: str, pasta_destino: str):
    """Baixa e extrai um repositório do GitHub na pasta de destino."""
    if not url_repositorio.endswith(".git"):
        url_repositorio = url_repositorio.rstrip("/") + ".git"

    repo_name = url_repositorio.split("/")[-1].replace(".git", "")
    owner = url_repositorio.split("/")[-2]

    print(f"Inciando processo de {owner}/{repo_name}")

    # Obtém o branch padrão via API do GitHub
    branch = obter_branch_padrao(owner, repo_name)
    if branch is None:
        print("Não foi possível determinar o branch padrão. Verifique a URL.")
        return

    zip_url = f"https://github.com/{owner}/{repo_name}/archive/refs/heads/{branch}.zip"

    response = requests.get(zip_url, stream=True)

    if response.status_code == 200:
        print("Download concluído. Extraindo os arquivos...")

        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(pasta_destino)

        print(f"Repositório '{repo_name}' baixado e extraído em: {pasta_destino}")
    else:
        print(f"Erro ao baixar o repositório. Código HTTP: {response.status_code}")
    print("-----------")

if __name__ == "__main__":
    destino = "/home/munif-gebara-junior/temp"

    if not os.path.exists(destino):
        os.makedirs(destino)

    baixar_repositorio_github("https://github.com/thingsboard/thingsboard", destino)
    baixar_repositorio_github("https://github.com/flowable/flowable-engine", destino)
    baixar_repositorio_github("https://github.com/spring-io/initializr", destino)
    baixar_repositorio_github("https://github.com/obsidiandynamics/kafdrop", destino)
    baixar_repositorio_github("https://github.com/aiven/klaw", destino)
    baixar_repositorio_github("https://github.com/corona-warn-app/cwa-server", destino)
    baixar_repositorio_github("https://github.com/cloudfoundry/uaa", destino)
    baixar_repositorio_github("https://github.com/apolloconfig/apollo", destino)
    baixar_repositorio_github("https://github.com/openzipkin/zipkin", destino)
    baixar_repositorio_github("https://github.com/spring-io/sagan", destino)

