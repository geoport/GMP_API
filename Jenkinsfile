pipeline {
  agent any
  stages {
    stage("Prune") {
      steps {
        sh "sudo docker system prune -f"
      }
    }
    stage("Deploy") {
      steps {
        sh "sudo docker-compose up --build -d"
      }
    }
  }
}
