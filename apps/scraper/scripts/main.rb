load "scripts/scraper_and_saver.rb"

puts "Enter Repository url: "
repo_url = gets

puts "Enter Pull Request number: "
pr_number = gets

pr_url = "#{repo_url.strip}/pull/#{pr_number.strip}"


scraper_and_saver pr_url
